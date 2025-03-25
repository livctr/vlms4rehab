from Levenshtein import distance as levenshtein_distance
from loguru import logger as eval_logger

from data.utils_strokerehab import VIDEO_DIR, LABEL_DIR, LabelUtils


def _get_primitives_score(pred, ref):
    """Normalized sequence similarity using Levenshtein distance"""
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
    
    return {
        "edit_score": edit_score,
        "action_error_rate": action_error_rate
    }


def sr_primitives_doc_to_visual(doc):
    return [VIDEO_DIR + doc["path_v"]]

def sr_primitives_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # TODO: Prompt for primitive counting task
    handedness = LabelUtils.get_handedness(LABEL_DIR + doc["path_l"])
    prompt = (
        f"Output 'REACH', 'TRANSPORT'. Focus on {handedness} hand.\n"
    )
    return prompt

def sr_primitives_doc_to_target(doc):
    csv_path = LABEL_DIR + doc["path_l"]
    handedness = LabelUtils.get_handedness(csv_path)
    action_seq = LabelUtils.convert_labels_to_action_sequence(csv_path, handedness)
    return action_seq

def sr_primitives_process_results(doc, results):
    """Process per-document results into metric format"""
    gt_labels_with_time = sr_primitives_doc_to_target(doc)  # DataFrame, annotated on every frame
    gt_labels_seq = [action for time, action in gt_labels_with_time]

    def _parse_result(res):
        return [x.strip().lower() for x in res.split(",")]
    pred_labels_seq = [action for res in results for action in _parse_result(res)]

    scores = _get_primitives_score(pred_labels_seq, gt_labels_seq)

    return {
        **doc,
        **scores,
        "pred": "[" + ",".join(pred_labels_seq) + "]",
        "gt": "[" + ",".join(gt_labels_seq) + "]",
    }
