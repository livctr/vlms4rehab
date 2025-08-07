from typing import Dict, Tuple, List
from collections import defaultdict
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


IA_VIDEO_QUESTIONS = None

def sr_ia_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [os.path.join(DataPaths.IA_CLIPPED_VIDEO_DIR, doc["path_v"])]


def _get_video_questions():
    """
    Load the questions DataFrame from the IA questions CSV file.
    Returns:
        pd.DataFrame: DataFrame with columns ['fm_video', 'question', 'answer']
    """
    questions_df = pd.read_csv(DataPaths.IA_QUESTIONS_PATH)

    FM_ITEM_TO_FM_RANGE = {
        3: (3, 8), 4: (3, 8), 5: (3, 8), 6: (3, 8), 7: (3, 8), 8: (3, 8),
        9: (9, 11), 10: (9, 11), 11: (9, 11),
        12: (12, 12), 13: (13, 13), 14: (14, 14), 15: (15, 15), 16: (16, 16), 17: (17, 17),
        18: (18, 18), 19: (19, 19), 20: (20, 20), 21: (21, 21), 22: (22, 22), 23: (23, 23),
        24: (24, 25), 25: (24, 25),
        26: (26, 26), 27: (27, 27), 28: (28, 28), 29: (29, 29), 30: (30, 30),
        31: (31, 33), 32: (31, 33), 33: (31, 33),
    }

    video_questions: Dict[Tuple[int, int, str], List[Dict]] = defaultdict(list)

    for _, row in questions_df.iterrows():
        fm_video = row["fm_video"]
        row.pop("fm_video")

        fm_item = int(fm_video.split('_')[0])
        fm_type = fm_video.split('_')[1]  # C or I
        fm_low = FM_ITEM_TO_FM_RANGE[fm_item][0]
        fm_high = FM_ITEM_TO_FM_RANGE[fm_item][1]

        video_questions[(fm_low, fm_high, fm_type)].append(row.to_dict())

    return video_questions


def sr_ia_doc_to_text(doc, lmms_eval_specific_kwargs=None, return_ids=False):

    # path_v: video path
    # patient: patient ID, e.g. S0001
    # fm_low: lowest FM item in this video
    # fm_high: highest FM item in this video
    # side_shown: L, R, LRT, or LRS
    # repetition_index: number of the repetition
    # duration_s: length of the video in seconds
    # side_affected: Left or Right

    # Get all questions relevant to this video: (fm_low, fm_high, side_shown) -> question
    global IA_VIDEO_QUESTIONS
    if IA_VIDEO_QUESTIONS is None:
        IA_VIDEO_QUESTIONS = _get_video_questions()
    
    if doc["side_shown"] in ["L", "R"]:
        fm_type = "I"  # individual
    else:
        fm_type = "C"  # concatenated

    questions_with_meta = IA_VIDEO_QUESTIONS[(doc["fm_low"], doc["fm_high"], fm_type)]
    questions = [q["question"] for q in questions_with_meta]
    ids = [str(q["qid"]) for q in questions_with_meta]

    # side_affected: replace "left video" and "right video" with the correct video
    # Originally, the video for the "left" hand is either on the left or top of the screen.
    # And the for "right" hand is either on the right or bottom of the screen.
    # The questions assume the "left video" is the affected hand.
    # Now we need to change the questions to refer to the affected hand.

    if doc['side_shown'] in ["L", "R"]:
        pass

    elif doc['side_shown'] in ["LRT", "LRS"]:

        referred_video_mapping = {"left video": "", "right video": ""}
        if doc["side_affected"] == "Left":
            if "T" in doc["side_shown"]:
                referred_video_mapping["left video"] = "top video"
                referred_video_mapping["right video"] = "bottom video"
            elif "S" in doc["side_shown"]:
                referred_video_mapping["left video"] = "left video"
                referred_video_mapping["right video"] = "right video"
            else:
                raise ValueError(f"Invalid side_shown: {doc['side_shown']}")
        elif doc["side_affected"] == "Right":
            if "T" in doc["side_shown"]:
                referred_video_mapping["left video"] = "bottom video"
                referred_video_mapping["right video"] = "top video"
            elif "S" in doc["side_shown"]:
                referred_video_mapping["left video"] = "right video"
                referred_video_mapping["right video"] = "left video"
            else:
                raise ValueError(f"Invalid side_shown: {doc['side_shown']}")
        else:
            raise ValueError(f"Invalid side_affected: {doc['side_affected']}")

        # Replace in questions
        _pattern = re.compile(r'\b(?:left video|right video)\b')
        for i, q in enumerate(questions):
            questions[i] = _pattern.sub(lambda m: referred_video_mapping[m.group(0)], q)

    else:
        raise ValueError(f"Invalid side_shown: {doc['side_shown']}")

    sep = " <SEP> "  # keeps track of which questions are being asked
    if return_ids:
        return sep.join(questions), sep.join(ids)
    else:
        return sep.join(questions)


def sr_ia_doc_to_target(doc):
    return ""  # We will evaluate these later


def sr_ia_process_results(doc, results):
    """Process per-document results into metric format"""
    _, ids = sr_ia_doc_to_text(doc, return_ids=True)  # For knowing which questions were asked
    return {"qids": ids}


class OutputToResultsFilter:

    def __init__(self):
        pass

    def apply(self, resps, docs):
        """
        Args:
            resps (List[List[str]]): A list where each element is a list of responses.
                                     It is assumed that the first element (i.e. responses[0])
                                     contains the string we need to process.
            docs: Additional document/context information (unused here).
        """
        resps_filtered = []
        for i in range(len(resps)):
            payload = resps[i][0][0]
            response, start_time, end_time = payload
            string = f"{response} <TIME> {start_time:.3f} - {end_time:.3f}"
            resps_filtered.append(string)
        return resps_filtered


def load_strokerehab_ia_dataset(
    patients: str = 'all',
    fm_items: str = 'all',
    reps: str = 'all',
    video_regex: str = None
) -> datasets.Dataset:
    """
    Loads the StrokeRehab IA dataset from a cleaned metadata CSV (with columns
    path_v, patient, fm_low, fm_high, side_shown, repetition_index, duration_s, side_affected)
    and applies AND-ed filters on patient IDs, FM items, and repetition—or, if video_regex
    is set, filters purely by that regex on path_v.
    """
    # --- load metadata ---
    df = pd.read_csv(DataPaths.IA_VIDEO_METADATA_PATH)
    
    # ensure numeric types
    df['fm_low']           = df['fm_low'].astype(int)
    df['fm_high']          = df['fm_high'].astype(int)
    df['repetition_index'] = df['repetition_index'].astype(int)

    # --- regex override ---
    if video_regex is not None:
        df = df[df['path_v'].str.contains(video_regex, regex=True)]
    else:
        # --- patient filter ---
        if patients != 'all':
            pats = {p.strip() for p in patients.split(',')}
            df = df[df['patient'].isin(pats)]

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

            # keep rows whose [fm_low, fm_high] overlaps req
            def overlaps(row):
                low, high = row['fm_low'], row['fm_high']
                # any requested fm in this row's range?
                return any(low <= r <= high for r in req)

            df = df[df.apply(overlaps, axis=1)]

        # --- reps filter ---
        if reps != 'all':
            if reps != 'first':
                raise ValueError("`reps` must be 'all' or 'first'")
            # for each (patient, fm_low, fm_high, side_shown), keep only lowest repetition_index
            grp = ['patient', 'fm_low', 'fm_high', 'side_shown']
            df['min_rep'] = df.groupby(grp)['repetition_index'].transform('min')
            df = df[df['repetition_index'] == df['min_rep']]
            df = df.drop(columns='min_rep')

    # --- build and return HF dataset ---
    dataset = datasets.Dataset.from_pandas(df.reset_index(drop=True))
    dataset_dict = datasets.DatasetDict({'test': dataset})
    return dataset_dict


strokerehab_load_ia_dataset_S0001_first = partial(load_strokerehab_ia_dataset, patients='S0001', fm_items='all', reps='first')
strokerehab_load_ia_dataset_S0001 = partial(load_strokerehab_ia_dataset, patients='S0001', fm_items='all', reps='all')
