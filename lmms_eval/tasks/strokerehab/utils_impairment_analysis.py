import shutil
from typing import Dict, Tuple, List
from collections import defaultdict
from functools import partial
import os
import re
import numpy as np

import datasets
import pandas as pd

from data.utils_strokerehab import DataPaths, FM_ITEM_TO_FM_RANGE


PATIENTS = 'C00011,S0005,S0001,S00021'
USE_VIEW = 'question_view'
IA_VIDEO_QUESTIONS = None

def sr_ia_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [os.path.join(DataPaths.IA_CLIPPED_VIDEO_DIR, doc["path_v"])]


def _get_video_questions(questions_path):
    """
    Load the questions DataFrame from the IA questions CSV file.
    Returns:
        pd.DataFrame: DataFrame with columns ['fm_video', 'question', 'answer']
    """
    questions_df = pd.read_csv(questions_path)

    video_questions: Dict[Tuple[int, int, str, str], List[Dict]] = defaultdict(list)

    for _, row in questions_df.iterrows():
        fm_video = row["fm_video"]
        row.pop("fm_video")

        fm_item = int(fm_video.split('_')[0])
        fm_low = FM_ITEM_TO_FM_RANGE[fm_item][0]
        fm_high = FM_ITEM_TO_FM_RANGE[fm_item][1]
        fm_laterality = fm_video.split('_')[1]  # B, A, or H
        fm_view = fm_video.split('_')[2]  # F or S. 
        video_questions[(fm_low, fm_high, fm_laterality, fm_view)].append(row.to_dict())

    return video_questions


def sr_ia_doc_to_text(doc, questions_path, return_ids=False):

    # path_v,patient,fm_low,fm_high,laterality,video_view,side_affected,is_annotated_view,is_question_view,duration

    # path_v: video path
    # patient: patient ID
    # fm_low: lowest FM item in this video
    # fm_high: highest FM item in this video
    # laterality: B, A, or H
    # video_view: F or S
    # side_affected: Left or Right
    # affected_video_loc: Top, Center, Left

    # Get all questions relevant to this video: (fm_low, fm_high, fm_laterality, fm_view) -> question
    global IA_VIDEO_QUESTIONS, USE_VIEW
    if IA_VIDEO_QUESTIONS is None:
        IA_VIDEO_QUESTIONS = _get_video_questions(questions_path)

    # Current bug: using 'question_view' gives us duplicates when more than 1 view for the same FM item
    # is specified in the questions CSV. Originally, we did not specify the "view", so when two views
    # are present, we run doc_to_text twice and get the same set of questions for the two views.
    # Adding doc['video_view'] solves this issue, but let's see if 'annotated_view' still works.

    # For annotated view, if the annotated view and question view differ, you should check both 'F' and 'S'
    # to see if they're in the dictionary.
    # Get all questions relevant to this video: (fm_low, fm_high, fm_laterality, fm_view) -> question
    if USE_VIEW == 'question_view':
        questions_with_meta = IA_VIDEO_QUESTIONS[(doc["fm_low"], doc["fm_high"], doc["laterality"], doc["video_view"])]
    else:
        questions_with_meta = []
        questions_with_meta.extend(IA_VIDEO_QUESTIONS[(doc["fm_low"], doc["fm_high"], doc["laterality"], 'F')])
        questions_with_meta.extend(IA_VIDEO_QUESTIONS[(doc["fm_low"], doc["fm_high"], doc["laterality"], 'S')])

    questions = [q["question"] for q in questions_with_meta]
    ids = [str(q["qid"]) for q in questions_with_meta]

    # For videos with both lateralities, the affected side is either 
    # at the top or left of the video.

    # side_affected: replace "left video" and "right video" with the correct video
    # Originally, the video for the "left" hand is either on the left or top of the screen.
    # And the for "right" hand is either on the right or bottom of the screen.
    # The questions assume the "left video" is the affected hand.
    # Now we need to change the questions to refer to the affected hand.

    if doc['affected_video_loc'] in ["Left", "Center"]:
        pass

    elif doc['affected_video_loc'] == "Top":
        referred_video_mapping = {"left video": "top video", "right video": "bottom video"}
        # Replace in questions
        _pattern = re.compile(r'\b(?:left video|right video)\b', re.IGNORECASE)
        for i, q in enumerate(questions):
            questions[i] = _pattern.sub(lambda m: referred_video_mapping[m.group(0)], q)

    else:
        raise ValueError("Invalid affected_video_loc")

    sep = " <SEP> "  # keeps track of which questions are being asked
    if return_ids:
        return sep.join(questions), sep.join(ids)
    else:
        return sep.join(questions)


def sr_ia_doc_to_target(doc):
    return ""  # We will evaluate these later in `postprocess`.


def sr_ia_process_results(doc, results, questions_path):
    """Process per-document results into metric format"""
    _, ids = sr_ia_doc_to_text(doc, questions_path, return_ids=True)  # For knowing which questions were asked
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
            resp_filtered = ""
            for j in range(len(resps[i][0])):
                payload = resps[i][0][j]
                response, start_time, end_time = payload
                resp_filtered += f"<RESP> {response} <TIME> {start_time:.3f}-{end_time:.3f} "
            resps_filtered.append(resp_filtered)
        return resps_filtered


def load_strokerehab_ia_dataset(
    patients: str = 'all',
    fm_items: str = 'all',
    video_regex: str = None,
    metadata_path: str = None,
    filter_by: str = 'question_view'
) -> datasets.Dataset:
    """
    Loads the StrokeRehab IA dataset from a cleaned metadata CSV. Applies AND-ed filters on
    patient IDs, FM items, and repetition—or, if video_regex is set, filters purely by that
    regex on path_v.

    Arguments:
    - patients
    - fm_items
    - video_regex
    - metadata_path
    - filter_by: either 'annotated_view' or 'question_view'.
    
    Expected Columns:
        path_v,patient,fm_low,fm_high,laterality,video_view,side_affected,is_annotated_view,
        affected_video_loc,is_question_view,duration
    """
    # --- load metadata ---
    df = pd.read_csv(metadata_path)

    # ensure numeric types
    df['fm_low']           = df['fm_low'].astype(int)
    df['fm_high']          = df['fm_high'].astype(int)

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
    
    # Filter by view angle
    col_name = "is_" + filter_by
    df = df[df[col_name]]

    # --- build and return HF dataset ---
    dataset = datasets.Dataset.from_pandas(df.reset_index(drop=True))
    dataset_dict = datasets.DatasetDict({'test': dataset})
    return dataset_dict


# PATIENTS = 'C00011,S00021'
# Method 1: simultaneous feed-in
# - Load the right videos
load_sria1_3_30 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='3-17,19-30',
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH1,
    filter_by=USE_VIEW
)
load_sria1_31_33 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='31-33',
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH1,
    filter_by=USE_VIEW
)
# - Use the right prompts (and attach the right qids)
sr_ia_doc_to_text1 = partial(sr_ia_doc_to_text, questions_path=DataPaths.IA_QUESTIONS_PATH1)
sr_ia_process_results1 = partial(sr_ia_process_results, questions_path=DataPaths.IA_QUESTIONS_PATH1)


# Method 2: individual
# - Load the right videos
load_sria2_3_30 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='13',  # 3-17,19-30
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH2,
    filter_by=USE_VIEW
)
load_sria2_31_33 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='31-33',
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH2,
    filter_by=USE_VIEW
)
# - Use the right prompts (and attach the right qids)
sr_ia_doc_to_text2 = partial(sr_ia_doc_to_text, questions_path=DataPaths.IA_QUESTIONS_PATH2)
sr_ia_process_results2 = partial(sr_ia_process_results, questions_path=DataPaths.IA_QUESTIONS_PATH2)

# Method 3
load_sria3_3_30 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='3-17,19-30',
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH3,
    filter_by=USE_VIEW
)
load_sria3_31_33 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='31-33',
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH3,
    filter_by=USE_VIEW
)
sr_ia_doc_to_text3 = partial(sr_ia_doc_to_text, questions_path=DataPaths.IA_QUESTIONS_PATH3)
sr_ia_process_results3 = partial(sr_ia_process_results, questions_path=DataPaths.IA_QUESTIONS_PATH3)

# Method 4
load_sria4_3_30 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='3-17,19-30',
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH4,
    filter_by=USE_VIEW
)
load_sria4_31_33 = partial(
    load_strokerehab_ia_dataset, patients=PATIENTS, fm_items='31-33',
    metadata_path=DataPaths.IA_VIDEO_METADATA_PATH4,
    filter_by=USE_VIEW
)
sr_ia_doc_to_text4 = partial(sr_ia_doc_to_text, questions_path=DataPaths.IA_QUESTIONS_PATH4)
sr_ia_process_results4 = partial(sr_ia_process_results, questions_path=DataPaths.IA_QUESTIONS_PATH4)


if __name__ == "__main__":
    ds = load_strokerehab_ia_dataset(
        patients=PATIENTS,
        fm_items='3-17,19-30',
        metadata_path=DataPaths.IA_VIDEO_METADATA_PATH3,
    )
    import pdb ; pdb.set_trace()
    for row in ds['test']:
        video_path = os.path.join(DataPaths.IA_CLIPPED_VIDEO_DIR, row['path_v'])
        # Save to "examples/"
        examples_dir = "examples/"
        os.makedirs(examples_dir, exist_ok=True)
        shutil.copy(video_path, examples_dir)
        print(video_path)