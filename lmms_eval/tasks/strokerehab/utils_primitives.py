import os
import re
import numpy as np

from Levenshtein import distance as levenshtein_distance
from loguru import logger as eval_logger

from data.utils_strokerehab import (
    DataPaths, LabelUtils, resps_to_string, string_to_resps,
    convert_motion_contact_to_primitives
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
    for primitive in LabelUtils.PRIMITIVES:
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
    mae_dict["count_truth"] = sum(ref.count(primitive) for primitive in LabelUtils.PRIMITIVES)
    mae_dict["count_pred"] = sum(pred.count(primitive) for primitive in LabelUtils.PRIMITIVES)

    return {
        "edit_score": edit_score,
        "action_error_rate": action_error_rate,
        **mae_dict,
    }

def sr_primitives_doc_to_visual(doc, lmms_eval_specific_kwargs=None):

    if "use_video_with_segmentations" not in lmms_eval_specific_kwargs:
        raise ValueError(
            "I thought this would work? Just like with doc_to_text."
        )

    use_video_with_segmentations = lmms_eval_specific_kwargs.get("use_video_with_segmentations", False)

    if use_video_with_segmentations:
        # Use video with segmentations
        tracked_name = doc["path_v"].split(".")[0] + "_tracked.mp4"
        return [os.path.join(DataPaths.SAM2_ANNOTATED_VIDEOS_PATH, tracked_name)]
    else:
        return [os.path.join(DataPaths.RAW_VIDEO_DIR, doc["path_v"])]


def sr_primitives_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    # Throw an error if the specific kwargs are not provided
    use_video_with_segmentations = lmms_eval_specific_kwargs["use_video_with_segmentations"]
    prompt = lmms_eval_specific_kwargs["prompt"]

    if use_video_with_segmentations:
        which_hand = "highlighted hand in RED"
    else:
        which_hand = LabelUtils.get_handedness(os.path.join(DataPaths.RAW_LABEL_DIR, doc["path_l"])).upper() + " hand"

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
    gt_primitives, gt_times = LabelUtils.convert_labels_to_prims_times(csv_path)
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
                prims, times = convert_motion_contact_to_primitives(
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
