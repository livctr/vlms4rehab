# log_factor_plots_v2.py
from __future__ import annotations

import json, os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
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
        ax.text(xmin, y + 0.38, row_name, fontsize=9)
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
            out_png = os.path.join(plot_dir, f"{uid}_factors.png")
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
    log_path = "logs/strokerehab_primitives_3/qwen2_5_vl_32b/Qwen__Qwen2.5-VL-32B-Instruct/20251009_192151_samples_strokerehab_primitives_3.jsonl"
    plot_dir = "./viz/smc_baseline/"

    plot_all_examples_from_log(
        log_path=log_path,
        plot_dir=plot_dir,
        label_to_color=label_to_color,
        write_metrics=True,   # adds metrics to title + writes <uid>_metrics.json
    )
