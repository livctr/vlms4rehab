// App.tsx
import React, { useEffect, useMemo, useRef, useState, ChangeEvent, DragEvent } from "react";
import "./App.css";

/*
  Video Annotator – reorganized layout (augmented)

  Columns:
  1) Left: Folder drop + "Videos in folder" list
  2) Middle: Hotkeys blurb + FM input + Save/Undo + FM list (numbers→names)
  3) Right: Video player + timeline (top) and scrollable annotation table (bottom)

  Updates per request:
  - CSV basename = the **video folder name** (derived from webkitRelativePath when available).
  - Each annotation stores the **video** it belongs to; table shows Video column; CSV includes it.
  - Switching videos **does not clear** existing annotations; further annotations use the new video name.
  - Removed the L/R "Sides checklist" panel.
  - Warn if typed FM label is not in the predefined list (confirm before adding).
*/

type AnnType = "s" | "e" | "t";
interface Annotation { type: AnnType; time: number; fmItem: string; video: string; }
interface ClipPair { s: Annotation; e: Annotation; }

const SEEK_DELTA = 0.1;
const fmt = (t: number) => t.toFixed(3);
const canonicalFmBase = (s: string) => s.trim().replace(/_/g, "-");
const isMp4 = (name: string) => name.toLowerCase().endsWith(".mp4");

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fmRef = useRef<HTMLInputElement | null>(null);

  // Folder videos & selection
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [selected, setSelected] = useState<File | null>(null);
  const [folderBase, setFolderBase] = useState<string>("annotations");

  // Playback / timeline
  const [duration, setDuration] = useState(0);
  const [now, setNow] = useState(0);

  // Annotations
  const [fmItem, setFmItem] = useState("");
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [editIx, setEditIx] = useState<number>(-1);
  const [editText, setEditText] = useState("");

  // FM list (numbers + names) loaded from /fm_list.txt
  const [fmRequiredBase, setFmRequiredBase] = useState<string[]>([]);
  const [fmNameMap, setFmNameMap] = useState<Record<string, string>>({});
  const fmRequiredWithSides = useMemo(() => {
    const out: string[] = [];
    fmRequiredBase.forEach((b) => { out.push(`${b}L`); out.push(`${b}R`); });
    return out;
  }, [fmRequiredBase]);

  const [fmListStatus, setFmListStatus] = useState<"idle"|"loading"|"ok"|"error">("idle");
  const FM_TXT_PATH = "/fm_list.txt";

  function parseFmList(text: string) {
    const bases: string[] = [];
    const names: Record<string, string> = {};
    const lines = text.split(/\r?\n|\//); // allow slash-separated groups
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      const parts = line.split(/[\t,|]+/);
      const baseRaw = (parts[0] ?? "").trim();
      if (!baseRaw) continue;
      const base = canonicalFmBase(baseRaw);
      const name = (parts.slice(1).join(" ").trim()) || base;
      if (!names[base]) bases.push(base);
      names[base] = name;
    }
    return { bases: Array.from(new Set(bases)), names };
  }

  const loadFmList = async () => {
    try {
      setFmListStatus("loading");
      const res = await fetch(FM_TXT_PATH, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const { bases, names } = parseFmList(text);
      setFmRequiredBase(bases);
      setFmNameMap(names);
      setFmListStatus("ok");
    } catch (e) {
      console.error(e);
      setFmRequiredBase([]);
      setFmNameMap({});
      setFmListStatus("error");
    }
  };
  useEffect(() => { loadFmList(); }, []);

  // Derived for CURRENT video only
  const currentVideoName = selected?.name || "";

  const clipOpen = useMemo(() => {
    if (!currentVideoName) return false;
    const seq = annotations
      .filter(a => a.video === currentVideoName && a.type !== "t")
      .sort((a, b) => a.time - b.time);
    let open = false;
    for (const a of seq) {
      if (a.type === "s") { if (open) return true; open = true; }
      else if (a.type === "e") { if (!open) return false; open = false; }
    }
    return open;
  }, [annotations, currentVideoName]);

  const clipPairs: ClipPair[] = useMemo(() => {
    if (!currentVideoName) return [];
    const seq = annotations
      .filter(a => a.video === currentVideoName && a.type !== "t")
      .slice()
      .sort((a, b) => a.time - b.time);
    const out: ClipPair[] = [];
    let lastS: Annotation | null = null;
    for (const a of seq) {
      if (a.type === "s") lastS = a;
      else if (a.type === "e" && lastS) { out.push({ s: lastS, e: a }); lastS = null; }
    }
    return out;
  }, [annotations, currentVideoName]);

  const unequalStartsEnds = useMemo(() => {
    if (!currentVideoName) return false;
    const ns = annotations.filter(a => a.video === currentVideoName && a.type === "s").length;
    const ne = annotations.filter(a => a.video === currentVideoName && a.type === "e").length;
    return ns !== ne;
  }, [annotations, currentVideoName]);

  const hasOverlap = useMemo(() => {
    const pairs = clipPairs.filter(p => p.s.fmItem === p.e.fmItem);
    for (let i = 0; i < pairs.length; i++) {
      const a = pairs[i];
      if (a.s.time >= a.e.time) return true;
      for (let j = i + 1; j < pairs.length; j++) {
        const b = pairs[j];
        if (b.s.fmItem !== a.s.fmItem) continue;
        const overlap = Math.max(a.s.time, b.s.time) < Math.min(a.e.time, b.e.time);
        if (overlap) return true;
      }
    }
    return false;
  }, [clipPairs]);

  // Completion map: base -> { L: boolean, R: boolean }
  const completion = useMemo(() => {
    const hasAtLeastOnePair = (label: string) => {
      // Count a completed s..e range per *video*. If any video has one, it's considered completed.
      const byVideo = new Map<string, Annotation[]>();
      annotations
        .filter(a => a.fmItem === label && a.type !== "t")
        .forEach(a => {
          const arr = byVideo.get(a.video) ?? [];
          arr.push(a);
          byVideo.set(a.video, arr);
        });

      for (const arr of byVideo.values()) {
        arr.sort((a, b) => a.time - b.time);
        let open = false;
        for (const a of arr) {
          if (a.type === "s") open = true;
          else if (a.type === "e" && open) return true; // found one s..e
        }
      }
      return false;
    };

    const baseMap: Record<string, { L: boolean; R: boolean }> = {};
    for (const b of fmRequiredBase) {
      baseMap[b] = {
        L: hasAtLeastOnePair(`${b}L`),
        R: hasAtLeastOnePair(`${b}R`),
      };
    }
    return baseMap;
  }, [annotations, fmRequiredBase]);

  const annotationsSortedForTable = useMemo(
    () =>
      annotations
        .map((ann, idx) => ({ ann, idx }))
        .sort((x, y) => {
          const byVideo = x.ann.video.localeCompare(y.ann.video, undefined, { numeric: true, sensitivity: "base" });
          if (byVideo !== 0) return byVideo;
          return x.ann.time - y.ann.time;
        }),
    [annotations]
  );

  // Video URL
  const videoURL = useMemo(() => (selected ? URL.createObjectURL(selected) : ""), [selected]);
  useEffect(() => () => { if (videoURL) URL.revokeObjectURL(videoURL); }, [videoURL]);

  // Events
  const onLoaded = () => setDuration(videoRef.current?.duration || 0);
  const onTime   = () => setNow(videoRef.current?.currentTime || 0);

  // Global hotkeys (operate on current video only)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName.toLowerCase();
      const inInput = tag === "input" || tag === "textarea";
      const vid = videoRef.current;

      if (e.key === "f" && !inInput) { e.preventDefault(); fmRef.current?.focus(); return; }
      if (e.key === "Escape" && inInput) { (e.target as HTMLElement).blur(); return; }
      if (!vid || !selected) return;

      if (e.key === " ") { e.preventDefault(); vid.paused ? vid.play() : vid.pause(); return; }
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const d = e.key === "ArrowLeft" ? -SEEK_DELTA : SEEK_DELTA;
        vid.currentTime = Math.min(Math.max(0, vid.currentTime + d), vid.duration);
        return;
      }
      if (e.key === "z" && e.ctrlKey) { e.preventDefault(); undo(); return; }

      if (inInput) return;
      if (!["s", "e", "t"].includes(e.key)) return;

      // Validate FM label against predefined list; warn (confirm) if not found
      const isPredefined = fmRequiredWithSides.length === 0 || fmRequiredWithSides.includes(fmItem);
      if (!fmItem) { vid.pause(); alert("Enter an FM label (e.g., 12L, 3-8R) before annotating."); return; }
      if (!isPredefined) {
        vid.pause();
        const cont = window.confirm(`${fmItem} is not in the predefined list. Continue anyway?`);
        if (!cont) return;
      }

      const t = vid.currentTime;
      const videoName = selected.name;

      if (e.key === "s") {
        if (hasOpenForLabel(fmItem, videoName)) { alert(`Open start exists for ${fmItem} in this video. Press 'e' to close it.`); return; }
        setAnnotations(prev => insertChronologically(prev, { type: "s", time: t, fmItem, video: videoName }));
        return;
      }
      if (e.key === "e") {
        const openIdx = indexOfLastOpenStart(fmItem, videoName);
        if (openIdx === -1) { alert(`No open start for ${fmItem} in this video. Press 's' first.`); return; }
        const lastStart = annotations[openIdx];
        if (t <= lastStart.time) { alert("End must be after start."); return; }
        const wouldOverlap = clipPairs
          .filter(p => p.s.fmItem === fmItem)
          .some(p => Math.max(p.s.time, lastStart.time) < Math.min(p.e.time, t));
        if (wouldOverlap) { alert("This clip overlaps an existing clip for the same FM label."); return; }
        setAnnotations(prev => insertChronologically(prev, { type: "e", time: t, fmItem, video: videoName }));
        return;
      }
      if (e.key === "t") {
        setAnnotations(prev => insertChronologically(prev, { type: "t", time: t, fmItem, video: videoName }));
        return;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [fmItem, selected, annotations, clipPairs, fmRequiredWithSides]);

  // Helpers (per video)
  const hasOpenForLabel = (label: string, videoName: string) => {
    const seq = annotations
      .filter(a => a.video === videoName && a.fmItem === label && a.type !== "t")
      .sort((a, b) => a.time - b.time);
    let open = false;
    for (const a of seq) { if (a.type === "s") open = true; else if (a.type === "e") open = false; }
    return open;
  };

  const indexOfLastOpenStart = (label: string, videoName: string) => {
    let openIdx = -1;
    annotations.forEach((a, i) => {
      if (a.video === videoName && a.fmItem === label) {
        if (a.type === "s") openIdx = i;
        if (a.type === "e") openIdx = -1;
      }
    });
    return openIdx;
  };

  const insertChronologically = (prev: Annotation[], a: Annotation) =>
    [...prev, a].sort((x, y) => x.time - y.time);

  // Folder intake
  const deriveFolderBase = (files: File[]) => {
    // Try to infer the top-level folder from webkitRelativePath; fallback to "annotations"
    const withRel = files.find(f => (f as any).webkitRelativePath);
    if (withRel) {
      const rel: string = (withRel as any).webkitRelativePath as string; // e.g. "MyFolder/video1.mp4"
      const top = rel.split("/")[0] || "annotations";
      return top;
    }
    return "annotations";
  };

  const takeFolderFiles = (fs: FileList | null) => {
    if (!fs?.length) return;
    const vids = Array.from(fs).filter(f => isMp4(f.name));
    vids.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    setFolderFiles(vids);
    setFolderBase(deriveFolderBase(vids));
    if (vids.length) maybeSwitchVideo(vids[0]);
  };
  const dropFolder = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); takeFolderFiles(e.dataTransfer.files); };
  const chooseFolder = (e: ChangeEvent<HTMLInputElement>) => takeFolderFiles(e.target.files);

  const maybeSwitchVideo = (f: File) => {
    if (selected?.name === f.name) return;
    setSelected(f);
    setDuration(0);
    setNow(0);
  };

  const removeAnn = (i: number) => setAnnotations(a => a.filter((_, ix) => ix !== i));
  const undo = () => setAnnotations(a => a.slice(0, -1));

  const commitEdit = () => {
    if (editIx === -1) return;
    setAnnotations(a => a.map((ann, idx) => (idx === editIx ? { ...ann, fmItem: canonicalFmBase(editText) } : ann)));
    setEditIx(-1); setEditText("");
  };

  // Save to CSV
  const saveCsv = async () => {
    if (!selected && folderFiles.length === 0) return;

    if (clipOpen) {
      const proceed = window.confirm("An open start has no end. Close it at video end time?");
      if (!proceed) return;
      if (videoRef.current && currentVideoName) {
        const t = videoRef.current.duration;
        openLabels(currentVideoName).forEach(label => {
          setAnnotations(prev => insertChronologically(prev, { type: "e", time: t, fmItem: label, video: currentVideoName }));
        });
      }
    }

    const present = new Set(annotations.map(a => canonicalFmBase(a.fmItem)));
    const missing = fmRequiredWithSides.filter(req => !present.has(canonicalFmBase(req)));

    if (fmRequiredWithSides.length > 0 && missing.length > 0) {
      const ok = window.confirm(
        `The following FM items are missing (need both L and R):\n\n${missing.join(", ")}\n\nProceed anyway?`
      );
      if (!ok) return;
    }

    // --- New semicolon-delimited CSV format ---
    const fmt2 = (t: number) => t.toFixed(2);

    // Group annotations by (video, fmItem), ignoring "t" marks
    const byVideoAndFm = new Map<string, Map<string, Annotation[]>>();
    annotations
      .filter(a => a.type !== "t")
      .forEach(a => {
        const byFm = byVideoAndFm.get(a.video) ?? new Map<string, Annotation[]>();
        const arr = byFm.get(a.fmItem) ?? [];
        arr.push(a);
        byFm.set(a.fmItem, arr);
        byVideoAndFm.set(a.video, byFm);
      });

    // Build CSV lines
    const lines: string[] = [];
    lines.push("video_path;fm_item;times");

    const sortedVideos = Array.from(byVideoAndFm.keys())
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

    for (const video of sortedVideos) {
      const byFm = byVideoAndFm.get(video)!;
      const sortedFms = Array.from(byFm.keys())
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

      for (const fm of sortedFms) {
        const seq = byFm.get(fm)!.slice().sort((a, b) => a.time - b.time);

        const pairs: string[] = [];
        let openS: Annotation | null = null;
        for (const a of seq) {
          if (a.type === "s") {
            openS = a;
          } else if (a.type === "e" && openS) {
            pairs.push(`s:${fmt2(openS.time)},e:${fmt2(a.time)}`);
            openS = null;
          }
        }

        if (pairs.length > 0) {
          lines.push(`${video};${fm};${pairs.join(",")}`);
        }
      }
    }

    const csv = lines.join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url  = URL.createObjectURL(blob);
    const aTag = document.createElement("a");
    const base = folderBase || "annotations";
    aTag.href = url;
    aTag.download = `${base}_annotations.csv`;
    aTag.click();
    URL.revokeObjectURL(url);
  };

  const openLabels = (videoName: string) => {
    const map: Record<string, number> = {};
    annotations.filter(a => a.video === videoName && a.type !== "t").forEach(a => {
      const k = a.fmItem; map[k] = (map[k] ?? 0) + (a.type === "s" ? 1 : -1);
    });
    return Object.keys(map).filter(k => map[k] === 1);
  };

  // UI
  return (
    <div className="app grid-3">
      {/* Left column: folder + video list */}
      <div className="col left">
        <h2 className="title">Video Temporal Annotator</h2>

        <div
          className="card dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={dropFolder}
          onClick={() => document.getElementById("folderInput")?.click()}
          title="Drag a folder or click to choose"
        >
          <div className="card-title">Folder of videos</div>
          <div className="muted">Drop a folder or click to select. Only <code>.mp4</code> are listed.</div>
          <input
            id="folderInput"
            type="file"
            // @ts-ignore non-standard but supported in Chromium
            webkitdirectory=""
            multiple
            accept="video/mp4"
            hidden
            onChange={chooseFolder}
          />
        </div>

        <div className="card">
          <div className="card-title">Videos in folder</div>
          <div className="video-list">
            {folderFiles.length === 0 ? (
              <div className="muted">No videos loaded yet.</div>
            ) : (
              folderFiles.map(f => (
                <button
                  key={f.name}
                  className={`video-item ${selected?.name === f.name ? "active" : ""}`}
                  onClick={() => maybeSwitchVideo(f)}
                  title={f.name}
                >
                  {f.name}
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Middle column: controls */}
      <div className="col middle">
        <div className="card">
          <div className="card-title">Hotkeys</div>
          <p className="muted small">
            <strong>Space</strong> play/pause • <strong>s</strong> start • <strong>e</strong> end • <strong>t</strong> keyframe •
            ←/→ ±0.1s • <strong>f</strong> focus FM • Esc blur
          </p>
        </div>

        <div className="card">
          <div className="row">
            <label className="label">FM:&nbsp;
              <input
                ref={fmRef}
                className="input"
                type="text"
                value={fmItem}
                onChange={(e) => setFmItem(canonicalFmBase(e.target.value))}
                placeholder="e.g. 12L or 3-8R"
              />
            </label>
            <button className="btn primary" onClick={saveCsv} disabled={!folderFiles.length}>Save CSV</button>
            <button className="btn" onClick={undo} disabled={annotations.length === 0}>Undo</button>
          </div>
        </div>

        <div className="card">
          <div className="card-title">FM list</div>
          <div className="muted small">Source: <code>/fm_list.txt</code></div>
          <div className={`status ${fmListStatus}`}>
            {fmListStatus === "loading" && "Loading…"}
            {fmListStatus === "ok" && `Loaded ${fmRequiredBase.length} base item(s).`}
            {fmListStatus === "error" && "Failed to load. Check the file exists."}
          </div>
          <div className="actions"><button className="btn" onClick={loadFmList}>Reload</button></div>

          {fmRequiredBase.length > 0 && (
            <div className="fm-map">
              <div className="fm-map-title">Numbers → Names</div>
                <div className="fm-map-list">
                  {fmRequiredBase.map((b) => {
                    const bothSides = (completion[b]?.L && completion[b]?.R) ?? false;
                    return (
                      <div className={`fm-row ${bothSides ? "done" : ""}`} key={b} title={fmNameMap[b] || b}>
                        <code className="fm-code">{b}</code>
                        <span className="fm-name">— {fmNameMap[b] || b}</span>
                      </div>
                    );
                  })}
                </div>
            </div>
          )}
        </div>
      </div>

      {/* Right column: video + scrollable table */}
      <div className="col right">
        <div className="video-wrap card">
          <video
            ref={videoRef}
            src={videoURL}
            className="video"
            controls
            onLoadedMetadata={onLoaded}
            onTimeUpdate={onTime}
          />
          {duration > 0 && (
            <div className="annot-bar">
              {clipPairs.map((p, idx) => (
                <div
                  key={`pair-${idx}-${p.s.time}-${p.e.time}`}
                  className="range"
                  style={{ left: `${(p.s.time / duration) * 100}%`, width: `${((p.e.time - p.s.time) / duration) * 100}%` }}
                  title={`${p.s.fmItem}: ${fmt(p.s.time)}–${fmt(p.e.time)}`}
                />
              ))}
              {annotations.filter(a => a.video === currentVideoName && a.type === "t").map((a, i) => (
                <div
                  key={`t-${i}-${a.time}`}
                  className="tmark"
                  style={{ left: `${(a.time / duration) * 100}%` }}
                  title={`t@${fmt(a.time)} (${a.fmItem})`}
                />
              ))}
            </div>
          )}
          <div className="time">Current time: {fmt(now)} s</div>
        </div>

        <div className="table card scrollable">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Video</th><th>Type</th><th>Time (s)</th><th>FM</th><th />
              </tr>
            </thead>
              <tbody>
                {annotationsSortedForTable.map(({ ann, idx: origIx }, rowNum) => (
                  <tr key={`${ann.video}-${ann.time}-${rowNum}`}>
                    <td>{rowNum + 1}</td>
                    <td title={ann.video}>{ann.video}</td>
                    <td>{ann.type}</td>
                    <td>{fmt(ann.time)}</td>
                    <td>
                      {editIx === origIx ? (
                        <input
                          autoFocus
                          className="input"
                          value={editText}
                          onChange={(e) => setEditText(canonicalFmBase(e.target.value))}
                          onBlur={commitEdit}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commitEdit();
                            if (e.key === "Escape") { setEditIx(-1); setEditText(""); }
                          }}
                        />
                      ) : (
                        <span
                          className="editable"
                          onClick={() => { setEditIx(origIx); setEditText(ann.fmItem); }}
                          title="Click to edit"
                        >
                          {ann.fmItem || <em className="placeholder">(empty)</em>}
                        </span>
                      )}
                    </td>
                    <td className="cell-actions">
                      <button className="btn danger ghost" onClick={() => removeAnn(origIx)}>×</button>
                    </td>
                  </tr>
                ))}
              </tbody>
          </table>
        </div>

        <div className="status-panel">
          {unequalStartsEnds && <p className="warn">⚠ Number of starts and ends are unequal.</p>}
          {hasOverlap && <p className="warn">⚠ Overlapping or inverted clips detected (same FM label).</p>}
          {!unequalStartsEnds && !hasOverlap && annotations.length > 0 && (
            <p className="ok">✔ Annotations look consistent for this video.</p>
          )}
        </div>
      </div>
    </div>
  );
}
