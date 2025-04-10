from Levenshtein import distance as levenshtein_distance
from loguru import logger as eval_logger

from data.utils_strokerehab import VIDEO_DIR, LABEL_DIR, LabelUtils


def _get_primitives_score(pred, ref):
    """Normalized sequence similarity using Levenshtein distance"""
    pred = [x.lower() for x in pred]
    ref = [x.lower() for x in ref]

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
        mae_dict[f"{primitive}_mae"] = mae
        maes.append(mae)

    if len(maes) == 0:
        avg_mae = 0.
    else:
        avg_mae = sum(maes) / len(maes)
    mae_dict["avg_mae"] = avg_mae
    
    return {
        "edit_score": edit_score,
        "action_error_rate": action_error_rate,
        **mae_dict,
    }

def sr_primitives_doc_to_visual(doc):
    return [VIDEO_DIR + doc["path_v"]]

def sr_primitives_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    handedness = LabelUtils.get_handedness(LABEL_DIR + doc["path_l"])
    prompt = (
        f"Focus on the patient's {handedness.upper()} hand. Output the sequence of functional \n"
        f"primitives performed by the patient's {handedness.upper()} hand as a comma-separated list.\n\n"
        "Functional primitives: \n"
        "- IDLE: hand is waiting\n"
        "- REACH: hand in motion with the purpose of contact with an object\n"
        "- REPOSITION: hand in motion with no contact at the endpoint\n"
        "- STABILIZE: hand steady to keep a target object still\n"
        "- TRANSPORT: hand in motion to convey an object in space\n"
        "ONLY OUTPUT THE COMMA-SEPARATED LIST OF FUNCTIONAL PRIMITIVES!\n\n"
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
    pred_labels_seq = results[0].split(',')
    scores = _get_primitives_score(pred_labels_seq, gt_labels_seq)
    return {
        **doc,
        **scores,
    }


class CombineParseDedupFilter:

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
            # Parse the first response string in each group.
            responses = resps[i][0]
            doc = docs[i]

            # log the action sequence
            csv_path = LABEL_DIR + doc["path_l"]
            handedness = LabelUtils.get_handedness(csv_path)
            action_seq = LabelUtils.convert_labels_to_action_sequence(csv_path, handedness)
            eval_logger.debug(f"Ground Truth: {str(action_seq)}")

            eval_logger.debug("Predicted: \n")
            pred_lines = []
            for i, resp in enumerate(responses):
                pred_lines.append(f"Chunk {i+1}: Predicted: {resp}")
            eval_logger.debug("\n".join(pred_lines))

            # Convert spliced response to one response
            responses_list = [x.split(',') for x in responses]
            responses = []
            for sublist in responses_list:
                for item in sublist:
                    if len(responses) > 0 and item == responses[-1]:
                        continue
                    responses.append(item.strip())

            responses = ",".join(responses)

            resps_filtered.append(responses)

        return resps_filtered
