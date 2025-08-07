"""
=============

Utility functions for extracting StrokeRehab-IA answers from a model log,
converting them into item-level predictions, attaching ground truth, and
computing aggregate evaluation metrics.

Quick example
-------------


>>> from pathlib import Path
>>> from postprocess.ia.score_from_log import (
...     extract_answers,
...     compute_fm_scores,
...     aggregate_fm_metrics,
... )

# 1.  Parse the streaming JSON-lines log produced by your model
>>> log_path = Path("model_output.jsonl")

# 2. Extract answers to each question
>>> ans_df   = extract_answers(log_path)          # patient x QID grid

# 3. Since each FM item may need to mask multiple questions, we use a separate
#    function to get FM item-level predictions.
>>> score_df = compute_fm_scores(output_log_path=log_path)

# 3.  Evaluate (optionally “segmenting” on items / patients)
>>> metrics = aggregate_fm_metrics(
...     score_df,
...     fm_items="3-8,11,12",          # ← only these FM items
...     patients="S0001,S0002",        # ← only these patients
... )
>>> print(metrics)
{'accuracy': 0.78, 'apd': 0.65, 'mtsd': 1.25}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Set

import numpy as np
import pandas as pd

from data.utils_strokerehab import DataPaths

# --------------------------------------------------------------------------- #
# 1.  Parsing the model-output log                                           #
# --------------------------------------------------------------------------- #



def _parse_resp_block(block: str) -> tuple[list[str], float | None]:
    """
    Split a single “<RESP> … <TIME> start-end” block into the list of answers
    *without* trailing whitespace/TIME and return the end-time (float) if present.
    """
    time_match = re.search(r"<TIME>\s*([\d.]+)-([\d.]+)", block)
    end_time = float(time_match.group(2)) if time_match else None
    answers_part = block.split("<TIME>")[0]  # drop the time segment entirely
    answers = [a.strip() for a in answers_part.split("<SEP>")]
    return answers, end_time


def _aggregate_answers(
    qid: int,
    raw_answers: list[str],
    end_times: list[float | None],
    qtype: str | None,
) -> str:
    """
    Reduce the list of answers for one qid into a single string according to the rules.
    """
    # Special handling for the “count-until” questions --------------------------
    if qid == 97:  # knee touches ⇒ stop at cumulative 4
        threshold = 4
    elif qid == 98:  # nose touches ⇒ stop at cumulative 5
        threshold = 5
    else:
        threshold = None

    if threshold is not None:
        cum = 0
        for ans, t in zip(raw_answers, end_times):
            try:
                cum += int(float(ans))  # tolerate “2.0”, etc.
            except ValueError:
                continue
            if cum >= threshold:
                # “best guess”: end of the segment where the target count is reached
                return f"{t:.3f}" if t is not None else "N/A"
        return "N/A"

    # --------------------------------------------------------------------------
    # Generic aggregations driven by question_type
    #
    if qtype == "rate":
        nums = [float(a) for a in raw_answers if re.fullmatch(r"-?\d+(\.\d+)?", a)]
        if nums:
            return str(int(round(np.mean(nums))))
        return "N/A"

    if qtype == "binary":
        # normalise to lower-case “yes” / “no”
        normed = [a.strip().lower() for a in raw_answers if a.strip()]
        if normed:
            return Counter(normed).most_common(1)[0][0]
        return "N/A"

    # Fallback for anything else
    return raw_answers[-1] if raw_answers else "N/A"


def extract_answers(
    output_log_path: str | Path | Iterable[str | Path],
    questions_csv_path: str | Path = DataPaths.IA_QUESTIONS_PATH,
) -> pd.DataFrame:
    """
    Accept one log path **or an iterable** of log paths.  Handles multi-<RESP>
    logs, aggregates per the spec, and guarantees a complete patient×QID grid.
    """
    paths = (
        [output_log_path]
        if isinstance(output_log_path, (str, Path))
        else list(output_log_path)
    )

    # ---------- Question metadata ---------- #
    qmeta = pd.read_csv(
        questions_csv_path, usecols=["qid", "fm_video", "question_type"]
    )
    qid2fm = {row.qid: int(row.fm_video.split("_")[0]) for row in qmeta.itertuples()}
    qid2qtype = {row.qid: row.question_type for row in qmeta.itertuples()}
    universe_qids = qmeta["qid"]

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int]] = set()

    # ---------- Read every JSONL log ---------- #
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)

                patient = rec["doc"]["patient"]
                qids_in_line = [int(x) for x in rec["qids"].split("<SEP>")]
                n_q = len(qids_in_line)

                # The new format may be a list or a str
                joined: str = rec["filtered_resps"]
                if isinstance(joined, list):
                    joined = "<SEP>".join(joined)

                # ------------------------------------------------------------------
                # Split into individual <RESP> blocks, in chronological order
                # ------------------------------------------------------------------
                blocks = [b.strip() for b in joined.split("<RESP>") if b.strip()]

                # Collect answers + end-times per question index
                answers_per_idx: list[list[str]] = [[] for _ in range(n_q)]
                end_times = []  # one per block (used only by qid 97 / 98)

                for blk in blocks:
                    ans_list, t_end = _parse_resp_block(blk)
                    if len(ans_list) != n_q:
                        raise ValueError(
                            f"Mismatch: {len(ans_list)=} vs {n_q=} in line:\n{blk[:120]}..."
                        )
                    for i, ans in enumerate(ans_list):
                        answers_per_idx[i].append(ans)
                    end_times.append(t_end)

                # ------------------------------------------------------------------
                # Aggregate answers question-wise
                # ------------------------------------------------------------------
                for idx, qid in enumerate(qids_in_line):
                    key = (patient, qid)
                    if key in seen_pairs:
                        raise ValueError(
                            f"Duplicate answer for patient={patient!r}, qid={qid}"
                        )
                    seen_pairs.add(key)

                    agg_ans = _aggregate_answers(
                        qid,
                        answers_per_idx[idx],
                        end_times,
                        qid2qtype.get(qid),
                    )
                    rows.append(
                        dict(
                            patient=patient,
                            qid=qid,
                            answer=agg_ans,
                            fm_item=qid2fm[qid],
                        )
                    )

    # ---------- Assemble patient × QID grid ---------- #
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["patient", "qid", "answer", "fm_item"])

    patients = df["patient"].unique()
    full_idx = pd.MultiIndex.from_product([patients, universe_qids], names=["patient", "qid"])
    df = df.set_index(["patient", "qid"]).reindex(full_idx).reset_index()
    df["fm_item"] = df["fm_item"].fillna(df["qid"].map(qid2fm)).astype(int)
    df["answer"] = df["answer"].astype("string")
    return df


# --------------------------------------------------------------------------- #
# 2.  Item-level scoring                                                     #
# --------------------------------------------------------------------------- #


def _score_single_row(row: pd.Series) -> tuple[bool, int | None]:
    """
    Decide whether this row yields a *predicted* score.

    Returns
    -------
    got_score : bool
    score     : int | None
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

    m = re.search(r"\b([012])\b", ans)  # rate-type question
    return (True, int(m.group(1))) if m else (False, None)


def compute_fm_scores(
    *,
    output_log_path: str | Path,
    questions_csv_path: str | Path = DataPaths.IA_QUESTIONS_PATH,
    gt_csv_path: str | Path = DataPaths.IA_SCORES_PATH,
    side_col: str = "Side of body affected",
    id_col: str = "Subject ID",
) -> pd.DataFrame:
    """
    End-to-end conversion of raw answers into
    *pred_score* + *gt_score*, including the rule that **FM-18** is inferred
    from items 15-17 when missing.

    Returns
    -------
    DataFrame with columns
        patient | fm_item | pred_score | gt_score
    """
    # ------------------------------------------------------------------ #
    # 1)  Answers + question metadata
    # ------------------------------------------------------------------ #
    ans_df = extract_answers(output_log_path, questions_csv_path)

    qmeta = pd.read_csv(
        questions_csv_path,
        usecols=["qid", "question_type", "binary_no_score", "binary_yes_score"],
    )
    merged = ans_df.merge(qmeta, on="qid", how="left")

    # ------------------------------------------------------------------ #
    # 2)  Per-item prediction
    # ------------------------------------------------------------------ #
    scored_rows: list[dict[str, Any]] = []
    for (patient, fm_item), grp in merged.groupby(["patient", "fm_item"]):
        if fm_item == 33:  # item-33 not scored
            score = np.nan
        else:
            grp = grp.sort_values("qid")
            score = next(
                (s for got, s in (_score_single_row(r) for _, r in grp.iterrows()) if got),
                np.nan,
            )
        scored_rows.append(
            {"patient": patient, "fm_item": fm_item, "pred_score": score}
        )

    df = (
        pd.DataFrame(scored_rows)
        .sort_values(["patient", "fm_item"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------ #
    # 3)  *Add* FM-18 when missing (dependency on 15-17)
    # ------------------------------------------------------------------ #
    new_rows: list[dict[str, Any]] = []
    for patient, grp in df.groupby("patient"):
        if (grp["fm_item"] == 18).any():
            continue  # already present

        scores_15_17 = grp.loc[grp["fm_item"].isin([15, 16, 17]), "pred_score"]
        all_two = (len(scores_15_17) == 3) and ((scores_15_17 == 2).all())
        new_rows.append(
            {"patient": patient, "fm_item": 18, "pred_score": 2 if all_two else 0}
        )

    if new_rows:
        df = (
            pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            .sort_values(["patient", "fm_item"])
            .reset_index(drop=True)
        )

    df["pred_score"] = df["pred_score"].astype("Int64")  # nullable int

    # ------------------------------------------------------------------ #
    # 4)  Attach ground truth
    # ------------------------------------------------------------------ #
    gt = pd.read_csv(gt_csv_path).set_index(id_col)

    def _lookup_gt(row: pd.Series) -> int | float:
        pid = row.patient
        if pid not in gt.index:
            return np.nan
        side_suffix = "R" if str(gt.at[pid, side_col]).strip().capitalize() == "Right" else "L"
        col = f"{row.fm_item}{side_suffix}"
        return gt.at[pid, col] if col in gt.columns else np.nan

    df["gt_score"] = df.apply(_lookup_gt, axis=1).astype("Int64")
    return df


# --------------------------------------------------------------------------- #
# 3.  Aggregate evaluation metrics                                           #
# --------------------------------------------------------------------------- #


def _parse_patients(spec: str | None) -> Set[str] | None:
    if spec is None or not str(spec).strip():
        return None
    return {p.strip() for p in spec.split(",") if p.strip()}


def _parse_items(spec: str | None) -> Set[int] | None:
    if spec is None or not str(spec).strip():
        return None
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if m := re.fullmatch(r"(\d+)-(\d+)", tok):
            start, end = map(int, m.groups())
            out.update(range(start, end + 1))
        elif tok.isdigit():
            out.add(int(tok))
        else:
            raise ValueError(f"Unparsable fm_item token: {tok!r}")
    return out


def _all_items_from_csv(csv_path: str | Path) -> Set[int]:
    qmeta = pd.read_csv(csv_path, usecols=["fm_video"])
    return {int(x.split("_")[0]) for x in qmeta["fm_video"]}


def aggregate_fm_metrics(
    score_df: pd.DataFrame,
    questions_csv_path: str | Path = DataPaths.IA_QUESTIONS_PATH,
    *,
    fm_items: str | None = None,
    patients: str | None = None,
) -> dict[str, float]:
    """
    Compute Accuracy, Average-Patient-Deviation (APD) and
    Mean-Total-Score-Deviation (MTSD) for (optionally) selected
    items and/or patients.
    """
    # 1) Item & patient subset
    item_set = _parse_items(fm_items) or _all_items_from_csv(questions_csv_path)
    patient_set = set(score_df["patient"].unique())
    if patients is not None:
        patient_set &= _parse_patients(patients)  # intersection

    if not item_set:
        raise ValueError("Item subset is empty.")
    if not patient_set:
        raise ValueError("Patient subset is empty.")

    # 2) Slice & ensure full grid
    df = score_df.loc[
        score_df["patient"].isin(patient_set) & score_df["fm_item"].isin(item_set),
        ["patient", "fm_item", "pred_score", "gt_score"],
    ]
    full_idx = pd.MultiIndex.from_product(
        [sorted(patient_set), sorted(item_set)],
        names=["patient", "fm_item"],
    )
    df = (
        df.set_index(["patient", "fm_item"])
        .reindex(full_idx)
        .reset_index()
    )

    pred, gt = df["pred_score"], df["gt_score"]

    # 3) Accuracy
    accuracy = float((pred == gt).sum(min_count=1)) / len(df)

    # 4) APD  (missing pred → error 2)
    df["err"] = [
        2 if pd.isna(p) else abs(int(p) - int(g)) for p, g in zip(pred, gt)
    ]
    apd = df.groupby("patient")["err"].sum().mean()

    # 5) MTSD  (patient-level totals + penalty for misses)
    agg = df.groupby("patient").agg(
        pred_sum=("pred_score", lambda s: s.fillna(0).sum()),
        gt_sum=("gt_score", "sum"),
        n_missing=("pred_score", lambda s: s.isna().sum()),
    )
    agg["patient_diff"] = (
        (agg["pred_sum"] - agg["gt_sum"]).abs() + 2 * agg["n_missing"]
    )
    mtsd = agg["patient_diff"].mean()

    return {"accuracy": accuracy, "apd": apd, "mtsd": mtsd}


# Example
if __name__ == "__main__":
    # log_path_1 = "logs/strokerehab_ia_1/bot_8f/20250808_035258_samples_strokerehab_ia_1.jsonl"
    log_path_2 = "logs/strokerehab_ia_2/bot_8f/20250808_035316_samples_strokerehab_ia_2.jsonl"
    ans_df = extract_answers([log_path_2])
    print(ans_df)
    # print(f"Extracted {len(ans_df)} answers.")
    # print(ans_df.head())
    # score_df = compute_fm_scores(output_log_path=)
    # print(f"Computed scores for {len(score_df)} (patient, item) pairs.")
    # print(score_df.head())
    # metrics = aggregate_fm_metrics(score_df)
    # print("Metrics:", metrics)
