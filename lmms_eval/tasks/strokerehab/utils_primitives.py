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
    handedness = LabelUtils.get_handedness(LABEL_DIR + doc["path_l"])

    # Is the patient's hand in contact with an object? transport, stabilize vs. reach, reposition, idle
    # Is the patient's hand moving an object?
    prompt = (
        f"Is the patient facing  \n"
        # f"Describe the actions done by the patient's {handedness} hand in this clip.\n"
    )
    # prompt = (
    #     "reach: movement with the purpose of contact with an object\n"
    #     "transport: movement to convey an object in space\n"
    #     "reposition: movement toward or from an object with no contact at the endpoint\n"
    #     "stabilize: minimal movement to keep a target object still\n"
    #     "idle: minimal movement to stand at the ready\n\n"
    #     f"Output the sequence of actions done by the patient's {handedness.upper()} hand as a comma-separated list.\n"
    #     # f"Example: REACH,TRANSPORT,REPOSITION,STABILIZE,IDLE"
    # )
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

    # def _parse_result(res):
    #     return [x.strip().lower() for x in res.split(",")]
    # pred_labels_seq = [action for res in results for action in _parse_result(res)]

    scores = _get_primitives_score(pred_labels_seq, gt_labels_seq)

    return {
        **doc,
        **scores,
    }


class CombineParseDedupFilter:
    """
    """
    def __init__(self):
        # No additional parameters are required.
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

            # Define helper functions for mapping the ground truth label
            def map_gt_label(label):
                if label in ['transport', 'reach', 'reposition']:
                    return f"Yes. {label}"
                elif label in ['stabilize']:
                    return f"No [minimal]. {label}"
                elif label in ['idle']:
                    return f"No. {label}"
                else:
                    return f"Unknown. {label}"

            def get_ground_truth_for_time(time, gt_seq):
                """
                Given a time and a sorted ground truth sequence (list of (timestamp, label)),
                returns the mapped ground truth (Yes./No./Unknown.) corresponding to the most recent event.
                """
                # Assume the ground truth sequence is sorted by time
                current_label = gt_seq[0][1]
                for t, label in gt_seq:
                    if t <= time:
                        current_label = label
                    else:
                        break
                return map_gt_label(current_label)

            # predicted sequence

            # Print the time, predicted response, and ground truth on new lines
            times = [0.48 * i for i in range(len(responses))]
            eval_logger.debug("Predicted (non-deduplicated): \n")
            pred_lines = []
            for time, resp in zip(times, responses):
                gt_output = get_ground_truth_for_time(time, action_seq)
                pred_lines.append(f"{time}: Predicted: {resp} | Ground Truth: {gt_output}")
            eval_logger.debug("\n".join(pred_lines))


            # convert spliced response to one response
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

