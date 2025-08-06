
from functools import partial
import os
import re
import numpy as np

from Levenshtein import distance as levenshtein_distance
from loguru import logger as eval_logger

import datasets
import pandas as pd

from data.utils_strokerehab import (
    DataPaths, PrimitiveLabelUtils, resps_to_string, string_to_resps,
    HEALTHY_PATIENTS,
    MILD_PATIENTS,
    MODERATE_PATIENTS,
    SEVERE_PATIENTS,
)


def _dedup(lst):
    """Deduplicate the adjacent elements in a list while preserving order."""
    deduped = []
    for item in lst:
        if not deduped or item != deduped[-1]:
            deduped.append(item)
    return deduped

def _get_primitives_score(pred, ref):
    """Normalized sequence similarity using Levenshtein distance"""
    pred = _dedup([x.lower() for x in pred])
    ref = _dedup([x.lower() for x in ref])

    max_len = max(len(pred), len(ref))
    if max_len == 0:
        edit_score = 100.
    else:
        edit_dist = levenshtein_distance(pred, ref)
        edit_score = (1 - (edit_dist / max_len)) * 100.
    
    if len(ref) == 0:
        action_error_rate = 0.
    else:
        action_error_rate = edit_dist / len(ref)
    
    # LabelUtils.PRIMITIVES
    mae_dict = {}
    maes = []
    for primitive in PrimitiveLabelUtils.PRIMITIVES:
        pred_cnt = pred.count(primitive)
        ref_cnt = ref.count(primitive)
        mae = abs(pred_cnt - ref_cnt)
        mae_dict[f"mae_{primitive}"] = mae
        mae_dict[f"count_{primitive}"] = ref_cnt
        maes.append(mae)

    if len(maes) == 0:
        avg_mae = 0.
    else:
        avg_mae = sum(maes) / len(maes)
    mae_dict["mae_avg"] = avg_mae
    mae_dict["count_truth"] = sum(ref.count(primitive) for primitive in PrimitiveLabelUtils.PRIMITIVES)
    mae_dict["count_pred"] = sum(pred.count(primitive) for primitive in PrimitiveLabelUtils.PRIMITIVES)

    return {
        "edit_score": edit_score,
        "action_error_rate": action_error_rate,
        **mae_dict,
    }

def sr_ia_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [os.path.join(DataPaths.IA_CLIPPED_VIDEO_DIR, doc["path_v"])]


def _parse_fm_item_from_path_v(path_v):
    """
    Given a string like "9-11" or "15", return a (start, end) tuple:
      parse_fm_range("9-11") -> (9, 11)
      parse_fm_range("15")   -> (15, 15)
    """
    m = re.search(r'_FM(\d+(?:_\d+)?)_', path_v)
    if not m:
        raise ValueError(f"No FM-range found in path_v: {path_v!r}")
    fm_str = m.group(1)
    parts = fm_str.split('_')
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    elif len(parts) == 1:
        v = int(parts[0])
        return v, v
    else:
        raise ValueError(f"Invalid FM string: {fm_str!r}")


def sr_ia_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    path_v = doc["path_v"]
    fm_low, fm_high = _parse_fm_item_from_path_v(path_v)

    # TODO: Put all of the questions here. Use <SEP>
    # Maybe also include video length?
    which_hand = "right hand"
    return (
        f"Focus on the patient's {which_hand}. Is it actively moving an object, "
        f"moving towards an object, or moving away from an object? Answer YES or NO.\n\n"
        f" <SEP> "
        f"Focus on the patient's {which_hand}. Is it actively grasping or holding an object?"
        f" Answer YES or NO.\n\n"
    )


def sr_ia_doc_to_target(doc):
    return ""  # We will evaluate these later


def sr_ia_process_results(doc, results):
    """Process per-document results into metric format"""
    # Extract the score from the results
    import pdb ; pdb.set_trace()  # Do stuff
    pred_primitives, pred_times = string_to_resps(results[0])

    # gt_string = sr_primitives_doc_to_target(doc)
    gt_primitives, gt_times = string_to_resps(gt_string)

    scores = _get_primitives_score(pred_primitives, gt_primitives)
    return {
        **doc,
        **scores,
    }


def flatten_resps(resps):
    """
    Args:
      resps: e.g. [[
               ('IDLE, REACH, REPOSITION, STABILIZE', 0.0, 0.84),
               ('IDLE - stuff, REACH - other random stuff, REPOSITION - yay!, STABILIZE - hiiii', 0.96, 1.8),
               ...
             ]]
    Returns:
      (primitives, times), where
        primitives = ['IDLE','REACH','REPOSITION','STABILIZE','IDLE',...]
        times      = [0.00, 0.28, 0.56, 0.84, 0.96,...]
    """
    VALID = {"IDLE", "REACH", "REPOSITION", "TRANSPORT", "STABILIZE"}
    # build regex once: \b(REACH|IDLE|...)\b, case-insensitive
    pattern = re.compile(r"\b(" + "|".join(VALID) + r")\b", flags=re.IGNORECASE)

    all_prims = []
    all_times = []

    # resps is a length-1 list containing one list of segments
    assert len(resps) == 1, "Expected resps to be a list with one element containing segments"
    for segment_list in resps:
        for text, start, end in segment_list:
            # find all occurrences of any VALID primitive
            matches = list(pattern.finditer(text))
            if not matches:
                continue

            # extract in-order, map to UPPER
            seg_prims = [m.group(1).upper() for m in matches]

            # linearly interpolate one timestamp per primitive
            ts = np.linspace(start, end, num=len(seg_prims), endpoint=False)
            all_prims.extend(seg_prims)
            all_times.extend(ts.tolist())
        all_times.append(end) # append the last time for the last segment

    return all_prims, all_times


def flatten_sep_resps_keep_full(resps):
    """
    Args:
      resps: [[
        ('Yes <SEP> Yes', start, end),
        ('Yes <SEP> Yes', start, end),
        ...
      ]]
    Returns:
      tokens:   ['Yes <SEP> Yes', 'Yes <SEP> Yes', ...]
      times:    [start0, start1, ..., end_last]
    """
    assert len(resps) == 1, "expected resps to be [[...]]"
    segments = resps[0]

    tokens = []
    times = []

    for text, start, end in segments:
        tokens.append(text)
        times.append(start)

    # append the final end-time to mark the end
    if segments:
        _, _, last_end = segments[-1]
        times.append(last_end)

    return tokens, times


def load_strokerehab_ia_dataset(
    patients: str = 'all',
    fm_items: str = 'all',
    reps: str = 'all',
    video_regex: str = None
) -> datasets.Dataset:
    """
    Loads the StrokeRehab IA (Instrumental Activities) dataset from a cleaned metadata file
    and applies AND-ed filters on patient IDs, FM items, and repetition—or, if video_regex
    is set, filters purely by that regex on the path.

    Args:
        patients: comma-sep list of patient IDs (e.g. "S0001,S0002"), or 'all' for no filter.
        fm_items: comma-sep list of fm numbers or ranges (e.g. "9,10-12,15"), or 'all' for no filter.
        reps: 'all' or 'first' — if 'first', keep only the lowest-numbered rep per video pattern.
        video_regex: a regex string; if provided, only rows where `path_v` matches this regex
                     will be kept, and all other filters are skipped.

    Returns:
        A HuggingFace Dataset of the filtered videos.
    """
    # --- load metadata ---
    df = pd.read_csv(DataPaths.IA_METADATA_PATH)  # must have column 'path_v'

    # --- regex override ---
    if video_regex is not None:
        df = df[df['path_v'].str.contains(video_regex, regex=True)]
    
    else:
        # --- patient filter ---
        if patients != 'all':
            pats = {p.strip() for p in patients.split(',')}
            df = df[df['path_v'].str.split('/', 1).str[0].isin(pats)]

        # --- fm_items filter ---
        if fm_items != 'all':
            # build set of requested fm numbers
            req = set()
            for token in fm_items.replace(' ', '').split(','):
                if '-' in token:
                    a, b = token.split('-', 1)
                    req.update(range(int(a), int(b) + 1))
                else:
                    req.add(int(token))

            # extract the fm part from each path
            df['__fm'] = df['path_v'].str.extract(r'_FM(\d+(?:-\d+)?)_')[0]

            def keep_fm(fm_str):
                if '-' in fm_str:
                    a, b = map(int, fm_str.split('-', 1))
                    return bool(req.intersection(range(a, b + 1)))
                else:
                    return int(fm_str) in req

            df = df[df['__fm'].apply(keep_fm)]
            df.drop(columns='__fm', inplace=True)

        # --- reps filter ---
        if reps != 'all':
            if reps != 'first':
                raise ValueError("`reps` must be 'all' or 'first'")
            # pull out rep number and base path (everything before final underscore)
            df['__rep'] = df['path_v'].str.extract(r'_(\d{2})\.mp4$')[0].astype(int)
            df['__base'] = df['path_v'].str.rsplit('_', 1).str[0]
            # keep the first (lowest) rep per base
            df = (
                df.sort_values('__rep')
                  .groupby('__base', as_index=False)
                  .first()
                  .drop(columns=['__rep', '__base'])
            )

    # --- build and return HF dataset ---
    return datasets.Dataset.from_pandas(df.reset_index(drop=True))


strokerehab_load_ia_dataset_S0001_first = partial(load_strokerehab_ia_dataset, patients='S0001', fm_items='all', reps='first')
strokerehab_load_ia_dataset_S0001 = partial(load_strokerehab_ia_dataset, patients='S0001', fm_items='all', reps='all')
