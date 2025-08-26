"""
postprocess.ia.eval
===================

Evaluation pipeline utilities for StrokeRehab IA logs.

This module exposes three utilities for extracting information from a log:
answers to questions, FM scores computed by the answers programatically, and
model evaluation scores.
  • answers_for_model(model, tasks=…, logs_root=…, log_paths=None, drop_parsed=True)
  • fm_scores_for_model(model, tasks=…, logs_root=…, log_paths=None, …)
  • metrics_for_model(model, tasks=…, logs_root=…, log_paths=None, …)
      → Calls _aggregate_fm_metrics under the hood

Multi-model convenience
-----------------------
  • metrics_for_models(models="all", tasks=…, logs_root=…, …)
      → Tidy DataFrame with metrics per model

Export
------
  • metrics_df_to_latex(df, caption=None, label=None, float_format="%.2f")
      → Convert metrics DataFrame to LaTeX table

Notes
-----
• Input JSONL logs are expected to contain:
    {
      "doc": {"patient": "S0001"},
      "qids": "0<SEP>1<SEP>2<SEP>…",
      "filtered_resps": "<RESP><TIME> 0.00-2.33 ... <SEP> ..."
    }
• QIDs < 95 → raw string answer
• QIDs 95–96 → rounded average of numeric answers
• QIDs 97–100 → elapsed time when cumulative count hits threshold
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from data.utils_strokerehab import DataPaths

from typing import Sequence

# ─────────────────────────── Constants & Regex ─────────────────────────── #

SEP_RE = re.compile(r"\s*<SEP>\s*")
RESP_RE = re.compile(r"\s*<RESP>\s*")
TIME_RE = re.compile(r"<TIME>\s*([\d.]+)-([\d.]+)", re.IGNORECASE)

# Thresholds for the “timing” questions (used by FM item 33 logic downstream).
TIMING_THRESHOLDS = {97: 4, 98: 5, 99: 4, 100: 5}

# QIDs that compute elapsed times (their answers are floats / np.inf).
TIMING_QIDS = set(TIMING_THRESHOLDS.keys())


# ─────────────────────────── Parsing helpers ─────────────────────────── #

def _parse_filtered_resps(
    raw: Union[str, List[str]]
) -> List[List[Tuple[str, float, float]]]:
    """
    Split `filtered_resps` into blocks, preserving (answer, t_start, t_end).

    Parameters
    ----------
    raw
        Either a single string or a list of strings that should be concatenated.

    Returns
    -------
    blocks : list[list[(answer, t_start, t_end)]]
        A list of blocks; each block corresponds to one <RESP> … segment and
        contains answers aligned to the QID order for that record.
    """
    if isinstance(raw, list):
        raw = "".join(raw)

    blocks: List[List[Tuple[str, float, float]]] = []
    for chunk in RESP_RE.split(raw):
        chunk = chunk.strip()
        if not chunk:
            continue

        m = TIME_RE.search(chunk)
        if not m:
            raise ValueError("Missing <TIME start-end> tag in a <RESP> block.")
        t_start, t_end = map(float, m.groups())

        # strip time portion and split answers by <SEP>
        answers_part = TIME_RE.sub("", chunk).strip()
        answers = [a.strip() for a in SEP_RE.split(answers_part) if a.strip()]
        blocks.append([(ans, t_start, t_end) for ans in answers])

    return blocks


def _calc_answer(qid: int, triples: List[Tuple[str, float, float]]) -> object:
    """
    Convert a list of (answer, t_start, t_end) for a single QID into a final answer.

    Returns
    -------
    object
        • For qid < 95: the raw (first) answer string (or pandas.NA).
        • For qid in {95,96}: the rounded average (as string) of numeric answers (or pandas.NA).
        • For qid in {97..100}: t_end (float) when cumulative count reaches threshold, else np.inf.
        • Otherwise: pandas.NA.
    """
    # 0–94 → echo first raw answer
    if qid < 95:
        return triples[0][0] if triples else pd.NA

    # 95,96 → average of numeric answers (rounded to nearest int)
    if qid in (95, 96):
        nums = []
        for ans, _, _ in triples:
            try:
                nums.append(float(ans))
            except (TypeError, ValueError):
                pass
        return str(int(round(np.mean(nums)))) if nums else pd.NA

    # 97–100 → time when cumulative count hits threshold
    if qid in TIMING_QIDS:
        threshold = TIMING_THRESHOLDS[qid]
        cum = 0
        for ans, _t0, t_end in triples:
            try:
                cum += int(float(ans))
            except (TypeError, ValueError):
                continue
            if cum >= threshold:
                return float(t_end)
        return np.inf

    # Fallback
    return pd.NA


# ───────────────────────────── Public API ───────────────────────────── #

def _extract_answers(
    output_log_path: Union[str, Path, Iterable[Union[str, Path]]],
    drop_parsed: bool = True,
) -> pd.DataFrame:
    """
    Parse one or more JSONL logs and produce a table of (patient, qid, answer).

    Parameters
    ----------
    output_log_path
        Path or iterable of paths to JSONL files. Each line must contain:
        - rec["doc"]["patient"] : str
        - rec["qids"]           : "q0<SEP>q1<SEP>..."
        - rec["filtered_resps"] : string with <RESP> blocks and <TIME> tags
    drop_parsed
        If True, omit the 'parsed_response' column (human-readable).

    Returns
    -------
    DataFrame
        Columns: patient | qid | answer
        (If drop_parsed=False, also includes parsed_response: a compact string
         encoding the per-block answers and start/end times.)
    """
    paths: List[Path]
    if isinstance(output_log_path, (str, Path)):
        paths = [Path(output_log_path)]
    else:
        paths = [Path(p) for p in output_log_path]

    # Pass 1 – collect all patients and qids seen
    patients: Set[str] = set()
    qids_seen: Set[int] = set()
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                patients.add(rec["doc"]["patient"])
                qids_seen.update(int(x) for x in SEP_RE.split(rec["qids"]))

    full_idx = pd.MultiIndex.from_product(
        [sorted(patients), sorted(qids_seen)], names=["patient", "qid"]
    )
    df = pd.DataFrame(index=full_idx).reset_index()
    df["answer"] = pd.NA
    df["parsed_response"] = pd.NA

    filled_pairs: Set[Tuple[str, int]] = set()

    # Pass 2 – compute parsed_response + final answer
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                patient = rec["doc"]["patient"]
                qids_line = [int(x) for x in SEP_RE.split(rec["qids"])]

                blocks = _parse_filtered_resps(rec["filtered_resps"])
                if any(len(b) != len(qids_line) for b in blocks):
                    raise ValueError(
                        f"Answer/QID mismatch for patient={patient} (block length != qids length)."
                    )

                # transpose: per-qid triples across blocks
                per_qid: List[List[Tuple[str, float, float]]] = [[] for _ in qids_line]
                for blk in blocks:
                    for idx, triple in enumerate(blk):
                        per_qid[idx].append(triple)

                for idx, qid in enumerate(qids_line):
                    key = (patient, qid)
                    if key in filled_pairs:
                        raise ValueError(f"Duplicate entry for (patient={patient}, qid={qid}).")
                    triples = per_qid[idx]

                    parsed = " | ".join(f"{ans} {ts:.3f}-{te:.3f}" for ans, ts, te in triples)
                    ans = _calc_answer(qid, triples)

                    mask = (df["patient"] == patient) & (df["qid"] == qid)
                    df.loc[mask, ["parsed_response", "answer"]] = [parsed, ans]
                    filled_pairs.add(key)

    if drop_parsed:
        df.drop(columns=["parsed_response"], inplace=True)

    # Keep stable dtypes
    df["patient"] = df["patient"].astype(str)
    df["qid"] = df["qid"].astype(int)
    return df

def _score_single_row(row: pd.Series) -> Tuple[bool, Optional[int]]:
    """
    Decide whether this *row* yields a usable predicted score and return it.

    For binary questions:
      - 'yes' → binary_yes_score (if present)
      - 'no'  → binary_no_score  (if present)

    For rate-type questions:
      - looks for a standalone 0/1/2 token in the answer string.

    Returns
    -------
    (got_score, score)
    """
    if pd.isna(row.answer):
        return False, None

    ans = str(row.answer).lower().strip()

    if row.question_type == "binary":
        if "yes" in ans and pd.notna(row.binary_yes_score):
            return True, int(row.binary_yes_score)
        if "no" in ans and pd.notna(row.binary_no_score):
            return True, int(row.binary_no_score)
        return False, None

    m = re.search(r"\b([012])\b", ans)
    return (True, int(m.group(1))) if m else (False, None)

def _compute_fm_scores_from_qids(
    ans_df: pd.DataFrame,
    *,
    questions_csv_path: Union[str, Path] = DataPaths.IA_QUESTIONS_PATH1,
    gt_csv_path: Union[str, Path] = DataPaths.IA_SCORES_PATH,
    side_col: str = "Side of body affected",
    id_col: str = "Subject ID",
) -> pd.DataFrame:
    """
    Convert the *answers* into predicted FM scores and attach ground truth.

    Parameters
    ----------
    ans_df
        Output of `_extract_answers` with columns: patient | qid | answer
    questions_csv_path
        CSV with columns at least:
          ['qid','fm_video','question_type','binary_no_score','binary_yes_score']
        Where 'fm_video' begins with the FM item id (e.g., "12_R_xxx" → 12).
    gt_csv_path
        Ground-truth CSV (wide) containing per-patient scores as columns
        like: '1L','1R','2L','2R',... and patient id column (id_col).
    side_col
        Column in GT CSV indicating "Left" or "Right" side affected.
    id_col
        Patient identifier column in GT CSV.

    Returns
    -------
    DataFrame
        Columns: patient | fm_item | pred_score | gt_score
        (pred_score, gt_score use pandas Int64 dtype)
    """
    # 1) Bring in question metadata
    qmeta = pd.read_csv(
        questions_csv_path,
        usecols=[
            "qid",
            "fm_video",
            "question_type",
            "binary_no_score",
            "binary_yes_score",
        ],
    ).copy()
    qmeta["fm_item"] = qmeta["fm_video"].str.split("_").str[0].astype(int)

    merged = ans_df.merge(qmeta, on="qid", how="left")

    # 2) Per-item prediction
    scored_rows: List[dict[str, Any]] = []
    for (patient, fm_item), grp in merged.groupby(["patient", "fm_item"], sort=True):
        if fm_item == 33:
            # Special: derive score from timing questions
            a_vals = grp.loc[grp["qid"].isin([97, 98]), "answer"].dropna()
            b_vals = grp.loc[grp["qid"].isin([99, 100]), "answer"].dropna()
            a = float(np.nanmin(a_vals.astype(float))) if not a_vals.empty else np.inf
            b = float(np.nanmin(b_vals.astype(float))) if not b_vals.empty else np.inf

            if np.isinf(a) or np.isinf(b):
                score = np.nan
            else:
                diff = a - b
                if diff < 2:
                    score = 2
                elif diff < 6:
                    score = 1
                else:
                    score = 0
        else:
            # take the first row that yields a score (binary or 0/1/2)
            grp = grp.sort_values("qid")
            score = next(
                (s for got, s in (_score_single_row(r) for _, r in grp.iterrows()) if got),
                np.nan,
            )

        scored_rows.append({"patient": patient, "fm_item": int(fm_item), "pred_score": score})

    df = (
        pd.DataFrame(scored_rows)
        .sort_values(["patient", "fm_item"])
        .reset_index(drop=True)
    )

    # 3) Infer FM-18 from 15–17 if missing:
    #     If items 15,16,17 all have pred_score == 2, set 18 := 2; else 0.
    to_add: List[dict[str, Any]] = []
    for patient, grp in df.groupby("patient", sort=False):
        present_items = set(grp["fm_item"].unique())
        # If 18 already present in this log, do nothing.
        if 18 in present_items:
            continue

        # Only infer 18 when 15,16,17 are all present for this patient in this log.
        if {15, 16, 17}.issubset(present_items):
            scores_15_17 = grp.loc[grp["fm_item"].isin([15, 16, 17]), "pred_score"]
            # All three must be non-missing to make a decision
            if scores_15_17.notna().sum() == 3:
                all_two = (scores_15_17.astype(int) == 2).all()
                inferred = 2 if all_two else 0
                to_add.append({"patient": patient, "fm_item": 18, "pred_score": inferred})
        # else: don’t synthesize FM-18 for this log

    if to_add:
        df = (
            pd.concat([df, pd.DataFrame(to_add)], ignore_index=True)
            .sort_values(["patient", "fm_item"])
            .reset_index(drop=True)
        )

    df["pred_score"] = df["pred_score"].astype("Int64")

    # 4) Attach ground truth
    gt = pd.read_csv(gt_csv_path).set_index(id_col)

    def _lookup_gt(row: pd.Series) -> Union[int, float]:
        pid = row.patient
        if pid not in gt.index:
            return np.nan
        side = str(gt.at[pid, side_col]).strip().capitalize()
        side_suffix = "R" if side == "Right" else "L"
        col = f"{row.fm_item}{side_suffix}"
        return gt.at[pid, col] if col in gt.columns else np.nan

    df["gt_score"] = df.apply(_lookup_gt, axis=1).astype("Int64")
    return df


def _parse_patients(spec: Optional[str]) -> Optional[Set[str]]:
    if spec is None or not str(spec).strip():
        return None
    return {p.strip() for p in spec.split(",") if p.strip()}


def _parse_items(spec: Optional[str]) -> Optional[Set[int]]:
    if spec is None or not str(spec).strip():
        return None
    out: Set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            start, end = map(int, m.groups())
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
        elif tok.isdigit():
            out.add(int(tok))
        else:
            raise ValueError(f"Unparsable fm_item token: {tok!r}")
    return out


def _all_items_from_csv(csv_path: Union[str, Path]) -> Set[int]:
    qmeta = pd.read_csv(csv_path, usecols=["fm_video"])
    return {int(x.split("_")[0]) for x in qmeta["fm_video"]}


def _aggregate_fm_metrics(
    score_df: pd.DataFrame,
    questions_csv_path: Union[str, Path] = DataPaths.IA_QUESTIONS_PATH1,
    *,
    fm_items: Optional[str] = None,
    patients: Optional[str] = None,
) -> dict[str, float]:
    """
    Compute Accuracy, Average-Patient-Deviation (APD), and Mean-Total-Score-Deviation (MTSD).

    Parameters
    ----------
    score_df
        Output of `_compute_fm_scores_from_qids` with columns:
        patient | fm_item | pred_score | gt_score
    questions_csv_path
        CSV used solely to infer the full set of fm_items when `fm_items` is not provided.
    fm_items
        Optional item spec like "3-8,9-11,12,13,14,15-18,19-33".
        Ranges and single integers allowed, comma-separated.
    patients
        Optional patient list like "S0001,S0002", comma-separated.

    Returns
    -------
    dict
        {'accuracy': float, 'apd': float, 'mtsd': float}
    """
    # 1) Item & patient subset
    item_set = _parse_items(fm_items) or _all_items_from_csv(questions_csv_path)
    patient_set = set(score_df["patient"].astype(str).unique())
    if patients:
        patient_set &= _parse_patients(patients)  # intersection

    if not item_set:
        raise ValueError("Item subset is empty.")
    if not patient_set:
        raise ValueError("Patient subset is empty.")

    # 2) Slice & ensure full grid
    df = score_df.loc[
        score_df["patient"].astype(str).isin(patient_set)
        & score_df["fm_item"].isin(item_set),
        ["patient", "fm_item", "pred_score", "gt_score"],
    ].copy()

    full_idx = pd.MultiIndex.from_product(
        [sorted(patient_set), sorted(item_set)],
        names=["patient", "fm_item"],
    )
    df = df.set_index(["patient", "fm_item"]).reindex(full_idx).reset_index()

    pred = df["pred_score"]
    gt = df["gt_score"]

    # 3) Accuracy (exact match rate). Denominator is the full grid size.
    accuracy = float((pred == gt).sum(min_count=1)) / len(df)

    # 4) APD – patient-wise sum of absolute per-item errors; missing pred ⇒ penalty 2.
    #    Then average across patients.
    df["err"] = [
        2 if pd.isna(p) or pd.isna(g) else abs(int(p) - int(g))
        for p, g in zip(pred, gt)
    ]
    apd = df.groupby("patient", dropna=False)["err"].sum().mean()

    # 5) MTSD – per-patient absolute diff of totals + 2×(#missing preds), averaged.
    agg = df.groupby("patient", dropna=False).agg(
        pred_sum=("pred_score", lambda s: s.fillna(0).sum()),
        gt_sum=("gt_score", "sum"),
        n_missing=("pred_score", lambda s: s.isna().sum()),
    )
    agg["patient_diff"] = (agg["pred_sum"] - agg["gt_sum"]).abs() + 2 * agg["n_missing"]
    mtsd = float(agg["patient_diff"].mean())

    return {"accuracy": float(accuracy), "apd": float(apd), "mtsd": float(mtsd)}


def _latest_log_path(task: str, model: str, *, logs_root: str | Path = "logs") -> Path:
    """Return the newest ``*_samples_*.jsonl`` log for ``task``/``model``.

    The directory layout may be either::

        logs/<task>/<model>/20250807_055937_samples_strokerehab_ia_1.jsonl

    or (with an extra run folder)::

        logs/<task>/<model>/<run_id>/20250807_055937_samples_strokerehab_ia_1.jsonl

    In the latter case the function automatically descends one level if – and
    only if – there is exactly **one** sub-directory below ``<model>``.
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

# --------------------------------------------------------------------------- #
# 2.  Model-level metrics                                                    #
# --------------------------------------------------------------------------- #

def _score_df_from_logs(
    log_paths: Iterable[str | Path],
    *,
    questions_csv_path: str | Path = DataPaths.IA_QUESTIONS_PATH1,
    gt_csv_path: str | Path = DataPaths.IA_SCORES_PATH,
) -> pd.DataFrame:
    """Helper: union of score DFs computed independently from several log files."""
    score_dfs: list[pd.DataFrame] = []
    for p in log_paths:
        ans_df = _extract_answers(Path(p))
        score_df = _compute_fm_scores_from_qids(
            ans_df,
            questions_csv_path=questions_csv_path,
            gt_csv_path=gt_csv_path,
        )
        score_dfs.append(score_df)
    return pd.concat(score_dfs, ignore_index=True) if score_dfs else pd.DataFrame(
        columns=["patient", "fm_item", "pred_score", "gt_score"]
    )

def answers_for_model(
    model: str,
    *,
    tasks: Sequence[str] = ("strokerehab_ia_1", "strokerehab_ia_2"),
    logs_root: str | Path = "logs",
    log_paths: Sequence[str | Path] | None = None,
    drop_parsed: bool = True,
) -> pd.DataFrame:
    """Like _extract_answers, but resolve *model* and *tasks* into log files if needed."""
    if log_paths is None:
        paths = [_latest_log_path(t, model, logs_root=logs_root) for t in tasks]
    else:
        paths = [Path(p) for p in log_paths]
    return _extract_answers(paths, drop_parsed=drop_parsed)


def fm_scores_for_model(
    model: str,
    *,
    tasks: Sequence[str] = ("strokerehab_ia_1", "strokerehab_ia_2"),
    logs_root: str | Path = "logs",
    log_paths: Sequence[str | Path] | None = None,
    questions_csv_path: str | Path = DataPaths.IA_QUESTIONS_PATH1,
    gt_csv_path: str | Path = DataPaths.IA_SCORES_PATH,
) -> pd.DataFrame:
    """Like _compute_fm_scores_from_qids, but resolve *model*/*tasks* logs first."""
    if log_paths is None:
        paths = [_latest_log_path(t, model, logs_root=logs_root) for t in tasks]
    else:
        paths = [Path(p) for p in log_paths]

    score_dfs = []
    for p in paths:
        ans_df = _extract_answers(p)
        score_df = _compute_fm_scores_from_qids(
            ans_df,
            questions_csv_path=questions_csv_path,
            gt_csv_path=gt_csv_path,
        )
        score_dfs.append(score_df)
    return pd.concat(score_dfs, ignore_index=True)
    

def metrics_for_model(
    model: str,
    *,
    tasks: Sequence[str] = ("strokerehab_ia_1", "strokerehab_ia_2"),
    logs_root: str | Path = "logs",
    log_paths: Sequence[str | Path] | None = None,
    questions_csv_path: str | Path = DataPaths.IA_QUESTIONS_PATH1,
    gt_csv_path: str | Path = DataPaths.IA_SCORES_PATH,
    fm_items: str | None = None,
    patients: str | None = None,
    **agg_kwargs,
) -> dict[str, float]:
    """Aggregate IA metrics for *model* across the chosen *tasks*.

    If *log_paths* is supplied it overrides *tasks*/*logs_root* and can point
    to arbitrary files.

    The *fm_items* and *patients* filters are passed through to
    :func:`_aggregate_fm_metrics` to “zoom in” on subsets.
    """
    if log_paths is None:
        paths = [_latest_log_path(t, model, logs_root=logs_root) for t in tasks]
    else:
        paths = [Path(p) for p in log_paths]

    score_df = _score_df_from_logs(
        paths, questions_csv_path=questions_csv_path, gt_csv_path=gt_csv_path
    )

    return _aggregate_fm_metrics(
        score_df,
        questions_csv_path=questions_csv_path,
        fm_items=fm_items,
        patients=patients,
        **agg_kwargs,
    )


# --------------------------------------------------------------------------- #
# 4.  Multi-model convenience                                                #
# --------------------------------------------------------------------------- #

def _list_models_with_logs(
    *, logs_root: str | Path, tasks: Sequence[str]
) -> list[str]:
    """Find models that have at least one matching log for any task."""
    models: set[str] = set()
    for t in tasks:
        task_dir = Path(logs_root) / t
        if not task_dir.is_dir():
            continue
        for model_dir in task_dir.iterdir():
            if not model_dir.is_dir():
                continue
            # Try direct logs first
            if any(model_dir.glob("*_samples_*.jsonl")):
                models.add(model_dir.name)
                continue
            # Or a single run subdir
            children = [p for p in model_dir.iterdir() if p.is_dir()]
            if len(children) == 1 and any(children[0].glob("*_samples_*.jsonl")):
                models.add(model_dir.name)
    return sorted(models)


def _parse_models_spec(
    models: str | Sequence[str] | None, *, logs_root: str | Path, tasks: Sequence[str]
) -> list[str]:
    """Turn 'all' or comma-separated string or sequence into a concrete list of model names."""
    if models is None or (isinstance(models, str) and models.strip().lower() == "all"):
        return _list_models_with_logs(logs_root=logs_root, tasks=tasks)
    if isinstance(models, str):
        return [m.strip() for m in models.split(",") if m.strip()]
    return list(models)


def metrics_for_models(
    models: str | Sequence[str] = "all",
    *,
    tasks: Sequence[str] = ("strokerehab_ia_1", "strokerehab_ia_2"),
    logs_root: str | Path = "logs",
    questions_csv_path: str | Path = DataPaths.IA_QUESTIONS_PATH1,
    gt_csv_path: str | Path = DataPaths.IA_SCORES_PATH,
    fm_items: str | None = None,
    patients: str | None = None,
    **agg_kwargs,
) -> pd.DataFrame:
    """Return a tidy DataFrame with metrics for each model.

    Parameters
    ----------
    models
        • "all" → discover all models under *logs_root* that have logs for any *tasks*
        • comma-separated string (e.g., "qwen2.5,internvl2")
        • sequence of model names
    tasks
        Iterable of IA task folder names under *logs_root* to consider.
    logs_root
        Root logs directory.
    questions_csv_path, gt_csv_path
        Passed through to scoring.
    fm_items, patients
        Optional filters forwarded to :func:`_aggregate_fm_metrics`.
    **agg_kwargs
        Any additional keyword args passed to :func:`_aggregate_fm_metrics`.

    Returns
    -------
    DataFrame
        Index = model, columns = ['accuracy', 'apd', 'mtsd'].
    """
    model_list = _parse_models_spec(models, logs_root=logs_root, tasks=tasks)
    if not model_list:
        raise ValueError("No models found for the given specification.")

    records: list[dict[str, float | str]] = []
    for m in model_list:
        res = metrics_for_model(
            m,
            tasks=tasks,
            logs_root=logs_root,
            questions_csv_path=questions_csv_path,
            gt_csv_path=gt_csv_path,
            fm_items=fm_items,
            patients=patients,
            **agg_kwargs,
        )
        res = dict(res)  # copy
        res["model"] = m
        records.append(res)

    df = pd.DataFrame.from_records(records).set_index("model").sort_index()
    # Ensure consistent column order
    return df[["accuracy", "apd", "mtsd"]]


# --------------------------------------------------------------------------- #
# 5.  LaTeX export                                                           #
# --------------------------------------------------------------------------- #

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


__all__ = [
    "answers_for_model",
    "fm_scores_for_model",
    "metrics_for_model",
    "metrics_for_models"
]


if __name__ == "__main__":
    MODEL = "qwen2_5_vl_7b"
    task1 = ("strokerehab_ia1_3_30", "strokerehab_ia1_31_33")  # simultaneous
    task2 = ("strokerehab_ia2_3_30", "strokerehab_ia2_31_33")  # individual
    print(answers_for_model(MODEL, tasks=task2))