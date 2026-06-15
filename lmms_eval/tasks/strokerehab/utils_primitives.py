
from functools import partial
import os
import re
import numpy as np

import datasets
import pandas as pd

from Levenshtein import distance as levenshtein_distance

from data.utils_strokerehab import (
    DataPaths, PrimitiveLabelUtils, resps_to_string, string_to_resps,
    HEALTHY_SUBJECTS,
    MILD_SUBJECTS,
    MODERATE_SUBJECTS,
    SEVERE_SUBJECTS,
)


def _dedup(lst):
    """Deduplicate the adjacent elements in a list while preserving order."""
    deduped = []
    for item in lst:
        if not deduped or item != deduped[-1]:
            deduped.append(item)
    return deduped

def _get_primitives_score(pred, ref, dedupe=True):
    """Normalized sequence similarity using Levenshtein distance"""
    if dedupe:
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

def sr_primitives_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [os.path.join(DataPaths.RAW_VIDEO_DIR, doc["path_v"])]


def sr_primitives_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    # Throw an error if the specific kwargs are not provided
    if lmms_eval_specific_kwargs is not None:
        use_video_with_segmentations = lmms_eval_specific_kwargs.get("use_video_with_segmentations", False)
        prompt = lmms_eval_specific_kwargs.get("prompt", "ideal")
    else:
        use_video_with_segmentations = False
        prompt = "ideal"

    if use_video_with_segmentations:
        which_hand = "highlighted hand in RED"
    else:
        which_hand = PrimitiveLabelUtils.get_handedness(os.path.join(DataPaths.RAW_LABEL_DIR, doc["path_l"])).upper() + " hand"

    if prompt == "ideal":
        # Use the original prompt without motion and contact
        return (
            f"Focus on the patient's {which_hand}. Output the sequence of functional "
            f"primitives performed by the patient's {which_hand} as a comma-separated list.\n\n"
            f"Functional primitives: \n"
            f"- IDLE: hand is waiting\n"
            f"- REACH: hand in motion with the purpose of contact with an object\n"
            f"- REPOSITION: hand in motion with no contact at the endpoint\n"
            f"- STABILIZE: hand steady to keep a target object still\n"
            f"- TRANSPORT: hand in motion to convey an object in space\n"
            f"Only output the functional primitives (no definitions) as a comma-separated list.\n\n"
        )
    elif prompt == "single_prediction":
        # Use the single prediction prompt without motion and contact
        return (
            f"Focus on the patient's {which_hand}. Output the functional primitive "
            f"performed by the patient's {which_hand} as a single word.\n\n"
            f"Functional primitives: \n"
            f"- IDLE: hand is waiting\n"
            f"- REACH: hand in motion with the purpose of contact with an object\n"
            f"- REPOSITION: hand in motion with no contact at the endpoint\n"
            f"- STABILIZE: hand steady to keep a target object still\n"
            f"- TRANSPORT: hand in motion to convey an object in space\n"
            f"Only output one functional primitive.\n\n"
        )
    elif prompt == "single_motion_and_contact":
        return (
            f"Focus on the patient's {which_hand}. Is it actively moving an object, "
            f"moving towards an object, or moving away from an object? Answer YES or NO.\n\n"
            f" <SEP> "
            f"Focus on the patient's {which_hand}. Is it actively grasping or holding an object?"
            f" Answer YES or NO.\n\n"
        )
    else:
        raise ValueError(
            f"Unknown prompt: {prompt}. Expected one of ['ideal', 'single_prediction', 'single_motion_and_contact']"
        )


def sr_primitives_doc_to_target(doc):
    csv_path = os.path.join(DataPaths.RAW_LABEL_DIR, doc["path_l"])
    gt_primitives, gt_times = PrimitiveLabelUtils.convert_labels_to_prims_times(csv_path)
    return resps_to_string(gt_primitives, gt_times)  # Ensure the format is correct


def sr_primitives_process_results(doc, results):
    """Process per-document results into metric format"""
    pred_primitives, pred_times = string_to_resps(results[0])

    gt_string = sr_primitives_doc_to_target(doc)
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


def _convert_motion_contact_to_primitives(
    motion_and_contact, times, future_window=2.0
):
    """
    Args:
        motion_and_contact: list of length n, each either
            - "Yes <SEP> Yes" strings, or
            - 2-tuples ("Yes"/"No", "Yes"/"No")
        times: list or tuple of floats of length n+1; times[i] is the start
            of segment i, times[i+1] its end.
        future_window: how many seconds ahead to scan for contact to label 'reach'
    
    Returns:
        primitives: list of str of length n, one of
            ["reach","reposition","transport","stabilize","idle"]
        times: the exact same list/tuple you passed in (length n+1)
    """
    n = len(motion_and_contact)
    assert len(times) == n + 1, "times must be one longer than motion_and_contact"

    # parse Yes/No into booleans
    motion_flags = []
    contact_flags = []
    for mc in motion_and_contact:
        if isinstance(mc, str):
            mot_str, con_str = mc.split("<SEP>")
            motion = "yes" in mot_str.strip().lower()
            contact = "yes" in con_str.strip().lower()
        else:
            motion = ("yes" in mc[0].strip().lower())
            contact = ("yes" in mc[1].strip().lower())
        motion_flags.append(motion)
        contact_flags.append(contact)

    primitives = []
    start_times = times[:-1]  # length n

    for i in range(n):
        t0 = start_times[i]
        m = motion_flags[i]
        c = contact_flags[i]

        if m and not c:
            # scan ahead up to future_window
            reach = False
            j = i + 1
            while j < n and (start_times[j] - t0) <= future_window:
                if contact_flags[j]:
                    reach = True
                    break
                j += 1
            prim = "reach" if reach else "reposition"

        elif m and c:
            prim = "transport"

        elif not m and c:
            prim = "stabilize"

        else:  # not m and not c
            prim = "idle"

        primitives.append(prim)

    # return the new primitives list, and the original times unchanged
    return primitives, times


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
            if "<SEP>" in resps[0][0][0][0]:
                motion_and_contact, times = flatten_sep_resps_keep_full(resps[i])
                prims, times = _convert_motion_contact_to_primitives(
                    motion_and_contact,
                    times,
                    future_window=2.0
                )
            else:
                prims, times = flatten_resps(resps[i])
                # Convert the flattened primitives and times into a string
            string = resps_to_string(prims, times)
            resps_filtered.append(string)
        return resps_filtered


def load_strokerehab_primitives_dataset(
        patients='all', activity='all', reps='all',
        filter_for_testset=False, filter_for_subsampled_testset=False,
        video_regex=None):
    """
    Loads the StrokeRehab dataset from a cleaned metadata file and applies AND-ed filters.

    Args:
        patients (str): The patient IDs to include in the dataset. Default is 'all'.
            If specifying individual patients, separate them with commas
            Example: 'S0001,S0002'
        activity (str): The activity names to include in the dataset. Default is 'all' (11 total).
            If specifying individual activities, separate them with commas
            Example: 'RTT left side,RTT right side,brushing,combing,deodrant,drinking,face wash,feeding,glasses,shelf left side,shelf right side'
        reps (str | int): Either 'all', 'first', or a number specifying the nth instance (includes repetition/view) to include.
        filter_for_testset (bool): If True, include only the VIDEOS in the test set of
            the original StrokeRehab paper. https://pubmed.ncbi.nlm.nih.gov/37766938/
            This data is located in './data/fp/strokerehab_test_set.txt' and already
            incorporated into the CSV metadata file.
        filter_for_subsampled_testset (bool): If True, include only the VIDEOS in the
            subsampled test set of the original StrokeRehab paper. 50 videos total. A subset of the
            original test set, which is ~515 videos.
        video_regex (str): A regex pattern to filter video paths. If provided, this will override
            the patient and activity filters. This is useful for loading specific videos based on their paths.
    
    Returns:
        dataset (datasets.Dataset): The StrokeRehab dataset with the specified filters applied.
    """
    df = pd.read_csv(DataPaths.FP_METADATA_PATH)
    if video_regex is not None:
        df = df[df['path_v'].str.contains(video_regex)]
    else:
        if patients != 'all':
            patients = patients.split(',')
            df = df[df['patient'].isin(patients)]
        if activity != 'all':
            activity = activity.split(',')
            df = df[df['activity'].isin(activity)]
        if reps != 'all':
            if reps != 'first':
                try:
                    reps = int(reps)
                except ValueError:
                    raise ValueError("Invalid value for reps. Must be 'all', 'first', or a number.")
            else:
                reps = 0
            df = df.sort_values('id').groupby(['patient', 'activity']).nth(reps).reset_index()
        if filter_for_testset:
            df = df[df['is_in_strokerehab_test_set']]
        if filter_for_subsampled_testset:
            df = df[df['subsampled_test_set']]
    dataset = datasets.Dataset.from_pandas(df)
    dataset_dict = datasets.DatasetDict({'test': dataset})
    return dataset_dict


strokerehab_load_dataset_C00015 = partial(load_strokerehab_primitives_dataset,
                                     patients='C00015',
                                     activity='brushing,combing,deodrant,drinking,feeding,glasses',
                                     reps='first')
test_videos = [
    "C00020/C00020_combing1_2.mkv",
    "C00020/C00020_shelf right side1_2.mkv",
]
regex = "|".join([f"({v})" for v in test_videos])
regex = rf"^({regex})$"
strokerehab_load_dataset_debug = partial(load_strokerehab_primitives_dataset, video_regex=regex)
strokerehab_load_dataset_S0001_small = partial(load_strokerehab_primitives_dataset, patients='S0001', reps='first')
strokerehab_load_dataset_S0001 = partial(load_strokerehab_primitives_dataset, patients='S0001')
strokerehab_load_dataset_3patients = partial(load_strokerehab_primitives_dataset, patients='C00011,S0001,S0002')
strokerehab_load_dataset_onerep = partial(load_strokerehab_primitives_dataset, reps='first')
strokerehab_load_dataset_test = partial(load_strokerehab_primitives_dataset, filter_for_testset=True)
strokerehab_load_dataset_test_subset = partial(load_strokerehab_primitives_dataset, filter_for_subsampled_testset=True)
strokerehab_load_dataset_healthy = partial(load_strokerehab_primitives_dataset, patients=HEALTHY_SUBJECTS)
strokerehab_load_dataset_mild = partial(load_strokerehab_primitives_dataset, patients=MILD_SUBJECTS)
strokerehab_load_dataset_moderate = partial(load_strokerehab_primitives_dataset, patients=MODERATE_SUBJECTS)
strokerehab_load_dataset_severe = partial(load_strokerehab_primitives_dataset, patients=SEVERE_SUBJECTS)
strokerehab_load_primitives_data = partial(load_strokerehab_primitives_dataset, patients='S0001', activity='brushing,combing', reps='first')

regex = r'^(C00020/C00020_glasses1_1.mkv|C00020/C00020_drinking1_1.mkv|C00020/C00020_combing1_1.mkv|C00020/C00020_face wash1_1.mkv|C00020/C00020_shelf right side1_1.mkv|C00020/C00020_deodrant1_1.mkv)$'
strokerehab_load_dataset_healthy_subset = partial(load_strokerehab_primitives_dataset, video_regex=regex)


strokerehab_load_small_test = partial(load_strokerehab_primitives_dataset,
                                      patients='C00020,C00023',
                                      reps=1)


def strokerehab_load_rtt_shelf_counting_dataset(test_set=True, patients_override=None):

    ds = load_strokerehab_primitives_dataset(activity='RTT left side,RTT right side,shelf left side,shelf right side')
    best_views = pd.read_csv('data/rs/best_views.txt')
    df = ds['test'].to_pandas()

    prompt_tune_patients = "C00020,C00023,S00019"
    # test_patients = (
    #     "C00022,C00024,C00025,C00028,C00029,"
    #     "S00013,S00016,S00023,S00010,S00012,"
    #     "S00011,S00017,S0001,S0003,S00018,"
    #     "S00021,S00029,S00031,S00034,S00049"
    # )
    # Override patient split either via argument or environment variable for easy job sharding.
    # Example:
    #   STROKEREHAB_COUNTING_PATIENTS="C00022,C00024,S00013" bash evaluate.sh ...
    env_patients_override = os.getenv("STROKEREHAB_COUNTING_PATIENTS", "").strip()
    if patients_override:
        patients = patients_override
    elif test_set:
        prompt_tune_set = set(prompt_tune_patients.split(','))
        patients = ",".join(p for p in df['patient'].unique() if p not in prompt_tune_set)
    else:
        patients = prompt_tune_patients
    if env_patients_override:
        patients = env_patients_override

    patients_list = [p.strip() for p in str(patients).split(',') if p.strip()]
    df = df[df['patient'].isin(patients_list)]

    # Optional bad-video exclusion to avoid full-run restarts caused by known problematic files.
    # This should be a regex over path_v entries.
    # Example:
    #   STROKEREHAB_COUNTING_EXCLUDE_VIDEO_REGEX="S00031/.+shelf right side"
    exclude_video_regex = os.getenv("STROKEREHAB_COUNTING_EXCLUDE_VIDEO_REGEX", "").strip()
    if exclude_video_regex:
        df = df[~df['path_v'].str.contains(exclude_video_regex, regex=True)]

    best_views_selected = pd.merge(df, best_views, on='id', how='inner')
    df = (best_views_selected
            .sort_values('id').groupby(['patient', 'activity'])
            .first()
            .reset_index()
    )
    df["activity_type"] = df["activity"].str.extract(r"(RTT|shelf)", expand=False)
    df = df.groupby(["patient", "activity_type"], as_index=False).first()
    df = df.drop(columns="activity_type")

    dataset = datasets.Dataset.from_pandas(df)
    dataset_dict = datasets.DatasetDict({'test': dataset})
    return dataset_dict


def load_strokerehab_final():
    # Same stroke patients
    ds = load_strokerehab_primitives_dataset(patients="S00013,S00016,S00023,S00011,S00017", reps="first")
    df_stroke = ds['test'].to_pandas()

    # C00026 -> C00027. Filter out double shelf/RTT exercises for control
    ds = load_strokerehab_primitives_dataset(patients="C00022,C00023,C00024,C00028,C00029", reps="first")
    df_control = ds['test'].to_pandas()
    df_control = df_control[~df_control['activity'].isin(['shelf left side', 'RTT left side'])]

    df_combined = pd.concat([df_stroke, df_control], ignore_index=True)

    # df_combined = df_combined.iloc[2:3]

    dataset = datasets.Dataset.from_pandas(df_combined)
    dataset_dict = datasets.DatasetDict({'test': dataset})
    return dataset_dict
