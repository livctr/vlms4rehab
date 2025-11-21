# log_factor_plots_v2.py
from __future__ import annotations

import re
import json, os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------
# Label normalization
# ----------------------------
_LABEL_ALIASES = {
    "idle": "IDLE",
    "reach": "REACH",
    "reposition": "REPOSITION",
    "stabilize": "STABILIZE",
    "transport": "TRANSPORT",
}

def _normalize_label(lbl: str) -> str:
    key = (lbl or "").strip().lower()
    return _LABEL_ALIASES.get(key, key.upper())

def _lower(lbl: str) -> str:
    return (lbl or "").strip().lower()

# ----------------------------
# Example wrapper
# ----------------------------
@dataclass
class Example:
    raw: Dict
    @property
    def uid(self) -> str:
        if "id" in self.raw and self.raw["id"]:
            return str(self.raw["id"])
        doc = self.raw.get("doc", {})
        bits = [doc.get("patient"), doc.get("activity"), doc.get("id") or self.raw.get("doc_id")]
        return "_".join([str(b) for b in bits if b])

    @property
    def duration(self) -> Optional[float]:
        doc = self.raw.get("doc", {})
        dur = doc.get("duration_s") or self.raw.get("duration_s")
        try:
            return float(dur) if dur is not None else None
        except (TypeError, ValueError):
            return None

# ----------------------------
# Parsing helpers
# ----------------------------
def _labelstarts_from_any(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    tokens = []
    def _walk(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                tokens.append(s)
        elif isinstance(x, (list, tuple)):
            for y in x:
                _walk(y)
    _walk(obj)
    tokens = [t for t in tokens if "@" in t]
    return ";".join(tokens)

def _starts_to_segments(pairs: List[Tuple[str, float]],
                        duration: Optional[float]) -> List[Tuple[str, float, float]]:
    if not pairs:
        return []
    collapsed: List[Tuple[str, float]] = []
    for lbl, t in pairs:
        if not collapsed or _normalize_label(lbl) != _normalize_label(collapsed[-1][0]):
            collapsed.append((lbl, float(t)))

    segs: List[Tuple[str, float, float]] = []
    for i, (lbl, start) in enumerate(collapsed):
        end = float(duration) if (i + 1 == len(collapsed) and duration is not None) else \
              float(collapsed[i + 1][1]) if (i + 1 < len(collapsed)) else start
        if duration is not None:
            end = min(end, float(duration))
        if end > start:
            segs.append((_normalize_label(lbl), float(start), float(end)))
    return segs

def _parse_labelstarts_string(s: str, duration: Optional[float]) -> List[Tuple[str, float, float]]:
    if not s:
        return []
    pairs: List[Tuple[str, float]] = []
    for token in s.split(";"):
        tok = token.strip()
        if not tok or "@" not in tok:
            continue
        lbl, ts = tok.split("@", 1)
        try:
            pairs.append((_normalize_label(lbl), float(ts)))
        except ValueError:
            continue
    return _starts_to_segments(pairs, duration)

def _parse_resps_nested(resps: List,
                        duration: Optional[float]) -> List[Tuple[str, float, float]]:
    segs: List[Tuple[str, float, float]] = []
    def _walk(o):
        if isinstance(o, list):
            for it in o:
                if isinstance(it, list) and len(it) == 3:
                    lbl, s, e = it
                    try:
                        lbl = _normalize_label(str(lbl))
                        s, e = float(s), float(e)
                        if duration is not None:
                            s = max(0.0, min(s, duration))
                            e = max(0.0, min(e, duration))
                        if e > s:
                            segs.append((lbl, s, e))
                    except Exception:
                        pass
                else:
                    _walk(it)
    _walk(resps)
    segs.sort(key=lambda x: x[1])
    merged: List[Tuple[str, float, float]] = []
    for lbl, s, e in segs:
        if not merged:
            merged.append((lbl, s, e))
        else:
            pl, ps, pe = merged[-1]
            if lbl == pl and abs(s - pe) < 1e-6:
                merged[-1] = (pl, ps, e)
            else:
                merged.append((lbl, s, e))
    return merged

# ----------------------------
# Public: record → DataFrames
# ----------------------------
def preds_and_truth_from_record(rec: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict]:
    ex = Example(rec)
    duration = ex.duration

    # Ground truth from "target"
    truth_str = _labelstarts_from_any(rec.get("target"))
    truth_segs = _parse_labelstarts_string(truth_str, duration)

    # Predictions from "filtered_resps" (preferred), else fall back to nested 'resps'
    pred_str = _labelstarts_from_any(rec.get("filtered_resps"))
    if pred_str:
        pred_segs = _parse_labelstarts_string(pred_str, duration)
    else:
        pred_segs = _parse_resps_nested(rec.get("resps", []), duration) if rec.get("resps") else []

    truth_df = pd.DataFrame(truth_segs, columns=["label", "start", "end"])
    pred_df = pd.DataFrame(pred_segs, columns=["label", "start", "end"])

    # metrics if present on the record
    meta = {
        "edit_score": rec.get("edit_score"),
        "action_error_rate": rec.get("action_error_rate"),
        "mae_avg": rec.get("mae_avg"),
    }
    return pred_df, truth_df, ex.uid, meta

# ----------------------------
# Plotting (consistent colors + no overlapping text)
# ----------------------------
def plot_factor_bars(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    title: str = "",
    save_path: Optional[str] = None,
    tlim: Optional[Tuple[float, float]] = None,
    label_to_color: Optional[Dict[str, str]] = None,  # lowercase label -> color
    write_metrics: bool = False,
    metrics: Optional[Dict] = None,
) -> None:
    """
    Draw GT and Pred rows with consistent colors. To avoid overlapping text,
    labels are drawn *only* on sufficiently wide segments.
    """
    # Determine bounds
    xmin = 0.0
    xmaxs: List[float] = []
    if not truth_df.empty:
        xmaxs += truth_df["end"].tolist()
    if not pred_df.empty:
        xmaxs += pred_df["end"].tolist()
    xmax = max(xmaxs) if xmaxs else 1.0
    if tlim:
        xmin, xmax = tlim
    total = max(1e-6, xmax - xmin)

    # Figure
    plt.figure(figsize=(11, 3.0))
    ax = plt.gca()

    def _bar_color(lbl: str) -> Optional[str]:
        if not label_to_color:
            return None
        return label_to_color.get(_lower(lbl))

    def _draw(df: pd.DataFrame, y: float, row_name: str):
        # background label on left
        ax.text(xmin, y + 0.38, row_name, fontsize=12)
        for _, r in df.iterrows():
            w = float(r["end"] - r["start"])
            if w <= 0:
                continue
            c = _bar_color(str(r["label"]))
            ax.barh(
                y, width=w, left=r["start"], height=0.35, align="center",
                edgecolor="black", linewidth=0.4, color=c
            )

    _draw(truth_df, y=1.0, row_name="GT")
    _draw(pred_df,  y=0.0, row_name="Pred")

    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (s)")

    # Title + metrics (optional)
    if write_metrics and metrics:
        es  = metrics.get("edit_score")
        aer = metrics.get("action_error_rate")
        metrics_str = []
        if es is not None:  metrics_str.append(f"Edit Score: {es:.2f}")
        if aer is not None: metrics_str.append(f"AER: {aer:.4f}")
        if metrics_str:
            title = f"{title}  —  " + " | ".join(metrics_str)
    if title:
        ax.set_title(title)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220)
        plt.close()
    else:
        plt.show()
        plt.close()

# ----------------------------
# Log readers
# ----------------------------
def iter_json_records_from_log(path: str) -> Iterable[Dict]:
    buf, brace_balance = [], 0
    def _maybe_emit():
        nonlocal buf, brace_balance
        if brace_balance == 0 and buf:
            blob = "\n".join(buf).strip()
            buf.clear()
            if not blob:
                return
            try:
                rec = json.loads(blob)
                yield rec
            except Exception:
                return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not buf and line.strip().startswith("{"):
                brace_balance += line.count("{") - line.count("}")
                buf.append(line)
                if brace_balance == 0:
                    for rec in _maybe_emit() or []:
                        yield rec
                continue
            # try single-line JSON
            if not buf:
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                    continue
                except Exception:
                    continue
            brace_balance += line.count("{") - line.count("}")
            buf.append(line)
            if brace_balance == 0:
                for rec in _maybe_emit() or []:
                    yield rec
    if brace_balance == 0:
        for rec in _maybe_emit() or []:
            yield rec

# ----------------------------
# Public APIs you asked for
# ----------------------------
def preds_to_df_from_log(log_path: str) -> List[Tuple[str, pd.DataFrame, pd.DataFrame, Dict]]:
    """Return [(uid, pred_df, truth_df, metrics_meta), ...] directly from the log."""
    outs: List[Tuple[str, pd.DataFrame, pd.DataFrame, Dict]] = []
    for rec in iter_json_records_from_log(log_path):
        try:
            pred_df, truth_df, uid, meta = preds_and_truth_from_record(rec)
            outs.append((uid, pred_df, truth_df, meta))
        except Exception:
            continue
    return outs

def plot_all_examples_from_log(
    log_path: str,
    plot_dir: str,
    label_to_color: Optional[Dict[str, str]] = None,
    write_metrics: bool = False,
) -> List[str]:
    """
    Parse each record in the log and save a factor-bar PNG per entry.
    If write_metrics=True, the image title includes Edit Score and AER,
    and a sidecar JSON (<uid>_metrics.json) is also written.
    """
    os.makedirs(plot_dir, exist_ok=True)
    saved: List[str] = []
    for rec in iter_json_records_from_log(log_path):
        try:
            pred_df, truth_df, uid, meta = preds_and_truth_from_record(rec)
            out_png = os.path.join(plot_dir, f"{uid}_factors.pdf")
            plot_factor_bars(
                pred_df,
                truth_df,
                title=uid,
                save_path=out_png,
                label_to_color=label_to_color,
                write_metrics=write_metrics,
                metrics=meta,
            )
            saved.append(out_png)

        except Exception:
            continue
    return saved




# --------------------------------
# NEW: multi-log comparison support
# --------------------------------
from dataclasses import dataclass

@dataclass
class MethodResult:
    method: str
    pred_df: pd.DataFrame
    truth_df: pd.DataFrame
    meta: Dict

def _safe_preds_from_log(log_path: str) -> List[Tuple[str, pd.DataFrame, pd.DataFrame, Dict]]:
    """
    Wrapper around preds_to_df_from_log with try/except isolation.
    """
    try:
        return preds_to_df_from_log(log_path)
    except Exception:
        return []

def collect_uid_index(log_specs: List[Tuple[str, str]]) -> Dict[str, Dict[str, MethodResult]]:
    """
    Build an index of uid -> { method_name: MethodResult } from multiple logs.

    Args:
        log_specs: List of (method_name, log_path)

    Returns:
        uid_index: Dict[uid, Dict[method_name, MethodResult]]
    """
    uid_index: Dict[str, Dict[str, MethodResult]] = {}
    for method, log_path in log_specs:
        for uid, pred_df, truth_df, meta in _reorder_preds_tuple(_safe_preds_from_log(log_path)):
            bucket = uid_index.setdefault(uid, {})
            bucket[method] = MethodResult(method=method, pred_df=pred_df, truth_df=truth_df, meta=meta or {})
    return uid_index

def _reorder_preds_tuple(rows: List[Tuple[str, pd.DataFrame, pd.DataFrame, Dict]]):
    """
    Your preds_to_df_from_log returns [(uid, pred_df, truth_df, meta)].
    Keep a helper to make the unpack explicit/robust.
    """
    for row in rows:
        uid, pred_df, truth_df, meta = row
        yield (uid, pred_df, truth_df, meta)

def _compute_global_tlim(method_results: List[MethodResult]) -> Tuple[float, float]:
    """
    Compute a common [xmin, xmax] across GT and Pred segments for all methods sharing a uid.
    If no segments, default to (0, 1).
    """
    xmin = 0.0
    xmaxs: List[float] = []
    for mr in method_results:
        if not mr.truth_df.empty:
            xmaxs += mr.truth_df["end"].tolist()
        if not mr.pred_df.empty:
            xmaxs += mr.pred_df["end"].tolist()
    xmax = max(xmaxs) if xmaxs else 1.0
    return (xmin, xmax)

def _bar_color_from_map(lbl: str, label_to_color: Optional[Dict[str, str]]) -> Optional[str]:
    if not label_to_color:
        return None
    return label_to_color.get((lbl or "").strip().lower())


def _draw_row(
    ax,
    df: pd.DataFrame,
    y: float,
    row_name: str,
    xmin: float,
    label_to_color: Optional[Dict[str, str]],
    *,
    bar_height: float = 0.78,      # 0..1 relative row thickness
    text_size: float = 14.0,       # label font size (scaled by your caller)
    label_dx: float = 0.02,        # label nudge to the right (in data units)
    edgecolor: Optional[str] = None,  # None for no edges; "black" for thin borders
    linewidth: float = 0.25,       # edge line width if edgecolor is set
) -> None:
    """
    Draw one horizontal row of segments and a left label.

    Notes:
    - `bar_height` is relative to the row spacing (we position rows on integer y's),
      so 0.65–0.85 is a good compact range.
    - `label_dx` nudges the row label slightly right from `xmin` to avoid colliding with bars.
    """
    # Row label (vertically centered on the row)
    ax.text(
        xmin + label_dx, y, row_name,
        fontsize=text_size, va="center", ha="left", weight="bold"
    )

    if df is None or df.empty:
        return

    for _, r in df.iterrows():
        start = float(r["start"])
        end   = float(r["end"])
        w = end - start
        if not np.isfinite(w) or w <= 0:
            continue

        c = _bar_color_from_map(str(r["label"]), label_to_color)

        ax.barh(
            y,
            width=w,
            left=start,
            height=bar_height,
            align="center",
            color=c,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2,
        )


def plot_uid_across_methods(
    uid: str,
    methods: List[MethodResult],
    *,
    save_path: Optional[str] = None,
    label_to_color: Optional[Dict[str, str]] = None,
    write_metrics: bool = False,
) -> None:
    """
    Create a single figure for one UID: top row is GT, followed by one Pred row per method.
    If a method is missing GT (shouldn't happen if logs are consistent), we still draw Pred rows.

    Row labels look like:
      - "GT"
      - "Pred — <method>  (ES=..., AER=...)" if write_metrics
    """
    if not methods:
        return

    # Compute a global time window
    xmin, xmax = _compute_global_tlim(methods)
    total_rows = 1 + len(methods)  # 1 GT + N methods

    plt.figure(figsize=(11, max(3.0, 1.2 * total_rows)))
    ax = plt.gca()

    # Draw GT using the *first available* truth_df (assumes same GT across methods)
    # We prefer a truth_df that isn't empty. If all empty, just skip GT row.
    gt_df = None
    for mr in methods:
        if not mr.truth_df.empty:
            gt_df = mr.truth_df
            break
    if gt_df is None and methods:
        # fallback: if all truth_df empty, build an empty frame to keep layout
        gt_df = pd.DataFrame(columns=["label", "start", "end"])

    # Y rows: GT at top
    y_gt = float(total_rows - 1)
    if gt_df is not None:
        _draw_row(ax, gt_df, y_gt, "GT", xmin, label_to_color,
          bar_height=0.75, text_size=14, label_dx=0.02,
          edgecolor=None, linewidth=0.25)

    # Methods (Pred rows)
    # Highest y at the top, then down
    for i, mr in enumerate(methods):
        y = float(total_rows - 2 - i)  # directly under GT
        # Label row, optionally with metrics
        row_name = f"Pred — {mr.method}"
        if write_metrics and isinstance(mr.meta, dict):
            es  = mr.meta.get("edit_score")
            aer = mr.meta.get("action_error_rate")
            suffix = []
            if es is not None:
                try:
                    suffix.append(f"ES={float(es):.2f}")
                except Exception:
                    suffix.append(f"ES={es}")
            if aer is not None:
                try:
                    suffix.append(f"AER={float(aer):.4f}")
                except Exception:
                    suffix.append(f"AER={aer}")
            if suffix:
                row_name += "  (" + ", ".join(suffix) + ")"

        _draw_row(ax, mr.pred_df, y, row_name, xmin, label_to_color,
          bar_height=0.75, text_size=13, label_dx=0.02,
          edgecolor=None, linewidth=0.25)

    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.set_title(uid, fontsize=20)
    ax.tick_params(axis='x', labelsize=20)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_all_examples_across_logs(
    log_specs: List[Tuple[str, str]],
    plot_dir: str,
    *,
    label_to_color: Optional[Dict[str, str]] = None,
    write_metrics: bool = False,
    only_overlap: bool = True,
) -> List[str]:
    """
    Compare multiple logs per item (uid). Writes one PNG per uid:
      <plot_dir>/<uid>_compare.pdf

    Args:
        log_specs: List of (method_name, log_path). The method_name appears on the Pred row.
        plot_dir: Output directory for PNGs.
        label_to_color: Optional dict lowercased label -> color hex (consistent with your single-plot).
        write_metrics: If True, each method row includes ES/AER when available.
        only_overlap: If True, plot only UIDs that appear in >= 2 of the provided logs.
                      If False, plot every UID present in at least one log (missing methods are just absent).

    Returns:
        List of saved PNG paths.
    """
    os.makedirs(plot_dir, exist_ok=True)
    uid_index = collect_uid_index(log_specs)

    # Determine overlap if requested
    # Build map: uid -> number of methods present
    uid_method_counts = {uid: len(mdict) for uid, mdict in uid_index.items()}

    saved: List[str] = []
    for uid, mdict in uid_index.items():
        if only_overlap and uid_method_counts.get(uid, 0) < 2:
            continue

        # Keep method order matching the input log_specs
        methods_in_order: List[MethodResult] = []
        seen = set()
        for method, _path in log_specs:
            mr = mdict.get(method)
            if mr is not None and method not in seen:
                methods_in_order.append(mr)
                seen.add(method)

        if not methods_in_order:
            continue

        out_png = os.path.join(plot_dir, f"{uid}_compare.pdf")
        plot_uid_across_methods(
            uid,
            methods_in_order,
            save_path=out_png,
            label_to_color=label_to_color,
            write_metrics=write_metrics,
        )
        saved.append(out_png)
    return saved


_TS_RE = re.compile(r"(?P<ts>\d{8}_\d{6})_samples_.*\.jsonl$")


def get_latest_sample_logs(base_log_dir: str | Path) -> List[Tuple[str, str]]:
    """
    For each immediate subfolder of `base_log_dir`, find the most recent file whose
    name matches:
        YYYYMMDD_HHMMSS_samples_<anything>.jsonl

    This searches recursively *within each* top-level subfolder to allow for
    structures like:
        <base>/<method>/**/YYYYMMDD_HHMMSS_samples_task.jsonl

    Returns:
        List of (subfolder_name, latest_log_path) pairs. Subfolders with no matching
        files are skipped.
    """
    base = Path(base_log_dir)
    if not base.exists() or not base.is_dir():
        return []

    results: List[Tuple[str, str]] = []

    # Only consider immediate children of base as "method" folders
    for method_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        latest_ts: Optional[pd.Timestamp] = None
        latest_path: Optional[Path] = None

        # Search recursively inside each method folder
        for root, _, files in os.walk(method_dir):
            for fname in files:
                # Fast pre-filter: we only care about files that look like samples jsonl
                if "_samples_" not in fname or not fname.endswith(".jsonl"):
                    continue

                m = _TS_RE.search(fname)
                if not m:
                    continue  # Requires a leading timestamp before `_samples_`

                ts_str = m.group("ts")
                try:
                    ts = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
                except Exception:
                    continue

                fpath = Path(root) / fname
                if (latest_ts is None) or (ts > latest_ts):
                    latest_ts = ts
                    latest_path = fpath

        if latest_path is not None:
            results.append((method_dir.name, str(latest_path)))

    return results


# ----------------------------
# Optional: small CLI
# ----------------------------

if __name__ == "__main__":
    label_to_color = {
        "idle": "#999999",       # dull gray  (153,153,153)
        "reach": "#009E73",      # bluish green (0,158,115)
        "transport": "#E69F00",  # orange (230,159,0)
        "reposition": "#56B4E9", # sky blue (86,180,233)
        "stabilize": "#CC79A7",  # purple (204,121,167)
    }

    models = [
        "internvl3p5_2b", "internvl3p5_8b", "internvl3p5_38b",
        "llava_next_video_7b", "llava_ov_7b",
        "nvila_8b",
        "qwen2_5_vl_7b", "qwen2_5_vl_32b", "qwen2_5_vl_72b",
    ]

    latest_log_specs = get_latest_sample_logs("logs/strokerehab_primitives_3/")
    latest_log_specs = {k: v for k, v in latest_log_specs}
    latest_log_specs["qwen2_5_vl_32b"] = "logs/strokerehab_primitives_3/qwen2_5_vl_32b/Qwen__Qwen2.5-VL-32B-Instruct/20251009_091138_samples_strokerehab_primitives_3.jsonl"

    filtered_log_specs = []
    for model in models:
        if model not in latest_log_specs:
            print(f"Warning: no log found for model '{model}'")
        else:
            filtered_log_specs.append((model, latest_log_specs[model]))

    saved = plot_all_examples_across_logs(
        log_specs=filtered_log_specs,
        plot_dir="./viz/all/",
        label_to_color=label_to_color,
        write_metrics=True,
        only_overlap=True,
    )
