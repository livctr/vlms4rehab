from __future__ import annotations

"""Utility helpers for evaluating StrokeRehab‑IA logs.

This module builds on :pymod:`postprocess.ia.score_from_log` and adds:

* **latest_log_path** – find the most recent ``.jsonl`` log under
  ``logs/<task>/<model>/``.
* **extract_answers_combined** – concat answers from multiple logs while
  guarding against duplicate *(patient, qid)* pairs.
* **metrics_for_model** – compute aggregate metrics for a *single* model,
  by default across both IA tasks.
* **metrics_for_models** – convenience wrapper returning a tidy
  :class:`~pandas.DataFrame` for several models.
* **metrics_df_to_latex** – render such a DataFrame into a LaTeX table.

Example
-------
>>> from postprocess.ia.metrics_utils import metrics_for_models
>>> df = metrics_for_models(["bot_8f", "bot_9g"])
>>> print(df)
>>> print(metrics_df_to_latex(df, caption="StrokeRehab‑IA results"))
"""

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from postprocess.ia.score_from_log import (
    compute_fm_scores,
    aggregate_fm_metrics,
)
import re

__all__ = [
    "latest_log_path",
    "extract_answers_combined",
    "metrics_for_model",
    "metrics_for_models",
    "metrics_df_to_latex",
]

# ---------------------------------------------------------------------------
# 1.  Locate log files                                                       #
# ---------------------------------------------------------------------------

def latest_log_path(task: str, model: str, *, logs_root: str | Path = "logs") -> Path:
    """Return the *newest* ``*_samples_*.jsonl`` log for *task*/*model*.

    The directory layout may be either::

        logs/<task>/<model>/20250807_055937_samples_strokerehab_ia_1.jsonl

    or (with an extra run folder)::

        logs/<task>/<model>/<run_id>/20250807_055937_samples_strokerehab_ia_1.jsonl

    In the latter case the function automatically descends one level if – and
    only if – there is exactly **one** sub-directory below ``<model>``.  This
    mirrors the heuristic previously used in :pyfunc:`get_latest_run_files`.

    The latest log is determined primarily via the embedded timestamp pattern
    ``YYYYMMDD_HHMMSS``; when absent, file *mtime* is used as a fallback.
    """
    base_dir = Path(logs_root) / task / model
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Drill down if there is exactly one sub-folder (run directory)
    children = [p for p in base_dir.iterdir() if p.is_dir()]
    search_dir = children[0] if len(children) == 1 else base_dir

    candidates = list(search_dir.glob("*_samples_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No *_samples_*.jsonl files in {search_dir}")

    ts_re = re.compile(r"(\d{8}_\d{6})_samples_")

    def _key(p: Path):
        m = ts_re.search(p.name)
        return m.group(1) if m else "00000000_000000"

    # Sort by timestamp string (lexicographically safe) – fallback to mtime
    candidates.sort(key=lambda p: (_key(p), p.stat().st_mtime), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# 2.  Combine answer grids safely                                           #
# ---------------------------------------------------------------------------

def extract_answers_combined(log_paths: Sequence[str | Path]):
    """Concatenate :func:`extract_answers` for many logs.

    Duplicate *(patient, qid)* pairs across logs raise a :class:`ValueError`.
    """
    from postprocess.ia.score_from_log import extract_answers  # local import to avoid circular deps

    dfs = [extract_answers(Path(p)) for p in log_paths]
    combined = pd.concat(dfs, ignore_index=True)

    dup_mask = combined.duplicated(subset=["patient", "qid"], keep=False)
    if dup_mask.any():
        offending = combined.loc[dup_mask, ["patient", "qid"]]
        raise ValueError(
            "Duplicate (patient, qid) pairs detected:\n" + offending.to_string(index=False)
        )
    return combined


# ---------------------------------------------------------------------------
# 3.  Model‑level metrics                                                   #
# ---------------------------------------------------------------------------

def _score_df_from_logs(log_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Helper: union of *score_df*s from several log files."""
    dfs = [compute_fm_scores(output_log_path=Path(p)) for p in log_paths]
    return pd.concat(dfs, ignore_index=True)


def metrics_for_model(
    model: str,
    *,
    tasks: Sequence[str] = ("strokerehab_ia_1", "strokerehab_ia_2"),
    logs_root: str | Path = "logs",
    log_paths: Sequence[str | Path] | None = None,
    **agg_kwargs,
) -> dict[str, float]:
    """Aggregate IA metrics for *model* across the chosen *tasks*.

    If *log_paths* is supplied it overrides *tasks*/*logs_root* and can point
    to arbitrary files.
    """
    if log_paths is None:
        paths = [latest_log_path(t, model, logs_root=logs_root) for t in tasks]
    else:
        paths = [Path(p) for p in log_paths]

    score_df = _score_df_from_logs(paths)
    return aggregate_fm_metrics(score_df, **agg_kwargs)


# ---------------------------------------------------------------------------
# 4.  Multi‑model convenience                                               #
# ---------------------------------------------------------------------------

def metrics_for_models(
    models: Sequence[str],
    *,
    tasks: Sequence[str] = ("strokerehab_ia_1", "strokerehab_ia_2"),
    logs_root: str | Path = "logs",
    **agg_kwargs,
) -> pd.DataFrame:
    """Return a tidy DataFrame with metrics for each *model*."""
    records = []
    for m in models:
        res = metrics_for_model(m, tasks=tasks, logs_root=logs_root, **agg_kwargs)
        res["model"] = m
        records.append(res)

    df = pd.DataFrame.from_records(records).set_index("model").sort_index()
    return df[["accuracy", "apd", "mtsd"]]


# ---------------------------------------------------------------------------
# 5.  LaTeX export                                                          #
# ---------------------------------------------------------------------------

def metrics_df_to_latex(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "%.2f",
) -> str:
    """Convert *df* (from :func:`metrics_for_models`) to a LaTeX table."""
    return df.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        column_format="lccc",
    )


if __name__ == "__main__":
    print(metrics_for_models(["llava_next_video_7b"]))