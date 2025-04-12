from typing import List, Tuple

from loguru import logger as eval_logger
from openai import OpenAI
import re
import torch
from transformers import pipeline
import yaml


from data.utils_strokerehab import VIDEO_DIR, ACTIVITY_GROUND_TRUTH_PATH

# Evaluation metrics
SUMMARY_STEPS_METRICS = ["precision", "recall", "f1", "ordering_score", "summary_steps_score"]

# Ground truth of the activity
try:
    with open(ACTIVITY_GROUND_TRUTH_PATH, "r") as file:
        agt = yaml.safe_load(file)
    activity2steps = {x.get("name"): x.get("steps") for x in agt}
except Exception as e:
    print(f"An error occurred while reading the YAML file: {e}")
    raise e

# Load LLM-as-a-judge
qwen2_llm_judge = None


def _init_llm_judge():
    global qwen2_llm_judge
    torch.cuda.empty_cache()
    qwen2_llm_judge = pipeline(
        task="text-generation",
        model="Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=0
    )


def _get_completion(prompt: str, 
                    max_new_tokens: int = 32,
                    ):
    global qwen2_llm_judge
    if not qwen2_llm_judge:
        _init_llm_judge()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    outputs = qwen2_llm_judge(messages,
                              max_new_tokens=max_new_tokens,
                              do_sample=False)

    return {
        "content": outputs[0]["generated_text"][-1]['content'],
    }


def _parse_summary_results(results: List[str]) -> List[str]:
    """
    Expects a list of list of steps. Parses them into a list of de-duplicated steps.

    For example,
    ```
    ['- remove glasses\n- wring the washcloth\n- twist the deodorant cap']
    ```

    Parses this into a list of strings. Throws error if format is incorrect.

    Example output: ["remove glasses", "wring the washcloth", "twist the deodorant cap"]
    """
    global qwen2_llm_judge
    if not qwen2_llm_judge:
        _init_llm_judge()

    # Remove any leading/trailing whitespace from the input.
    results = [result.split('\n-') for result in results]
    results = [item for sublist in results for item in sublist]
    results = [result.strip(" \t\n\r\f\v-") for result in results]
    if not results:
        return results

    # De-duplicate the adjacent items in the list using LLM-as-a-judge
    prompt_pre = (
        "Answer `yes` if statement 1 and statement 2 are semantically the same, otherwise `no`.\n\n"
    )

    dedup_results = [results[0]]

    for i in range(1, len(results)):
        current_step = results[i]
        last_added_step = dedup_results[-1]

        # Skip empty steps
        if not current_step:
            continue

        # Construct prompt to check if current step is the same as the last added step
        prompt = prompt_pre + f"Statement 1: {last_added_step}\nStatement 2: {current_step}"
        response = _get_completion(prompt, max_new_tokens=10)

        # Only add the step if it's not a duplicate (response is not "yes")
        if "yes" not in response["content"].lower():
            dedup_results.append(current_step)

    return dedup_results


def _get_bipartite_matching(pred_steps: List[str], gt_steps: List[str]) -> List[Tuple[int, int]]:
    """Get bipartite matching between predicted and ground truth steps."""
    # Deduplicate gt_steps
    deduped_gt_steps = dict()  # string to list of original indices
    for gt_step_idx, gt_step_str in enumerate(gt_steps):
        if gt_step_str not in deduped_gt_steps:
            deduped_gt_steps[gt_step_str] = [gt_step_idx]
        else:
            deduped_gt_steps[gt_step_str].append(gt_step_idx)
    deduped_keys = sorted(deduped_gt_steps.keys())  # string list of ground truth steps
    
    # 1-indexed!
    numbered_steps_str = "\n".join(f"{i+1}. {step}" for i, step in \
                                enumerate(deduped_keys))
    
    cache = {}  # string to deduplicated key indices (in case of multiple identical pred_steps)
    matches = []

    for pred_step_idx, pred_step_str in enumerate(pred_steps):
        prompt = (
            "Task: return the number(s) of the actions under `ACTIONS:` with the same "
            "semantic meaning as the query under `QUERY:` as a comma-separated string. "
            "For example, you might return \"1,2,3,4,5\". "
            "If no actions match, return `None`. \n\n"
            f"QUERY:\n{pred_step_str}"
            f"ACTIONS:\n{numbered_steps_str}\n\n"
        )
        response = _get_completion(prompt)

        # Create bipartite match graph
        if pred_step_str in cache:
            matches_deduped_idx = cache[pred_step_str]
        else:
            matches_deduped_idx = []
            if response["content"] != "None":
                try:
                    for x in response["content"].split(","):
                        x = int(re.sub(r'[^0-9]', '', x)) - 1
                        if 0 <= x < len(deduped_keys):
                            matches_deduped_idx.append(x)
                        else:
                            eval_logger.warning(f"Prediction to Ground Truth Step Index {x} "
                                                "out of range [0,{len(deduped_keys)-1}).. Ignored.")
                except Exception as e:
                    eval_logger.error(f"Error parsing response: {response['content']}")
                    eval_logger.error(f"Error: {e}")
            cache[pred_step_str] = matches_deduped_idx

        for match_deduped_idx in matches_deduped_idx:
            for original_gt_index in deduped_gt_steps[deduped_keys[match_deduped_idx]]:
                matches.append((pred_step_idx, original_gt_index))

    return matches

def _get_greedy_matches(
    semantic_matches: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    From a semantic match bipartite graph, greedily match the predictions
    to the ground truths. Assumes predicted on left, ground truth on right.

    O(n^2) time complexity, thinking n less than or around 10
    """

    # Filter for predictions and ground truths that have at least one match
    preds = []
    pred2gt = {}
    for pred, gt in semantic_matches:
        if pred not in preds:
            preds.append(pred)
        if pred not in pred2gt:
            pred2gt[pred] = [gt]
        else:
            if gt not in pred2gt[pred]:
                pred2gt[pred].append(gt)
    for pred in pred2gt:
        pred2gt[pred] = sorted(pred2gt[pred])
    preds = sorted(preds)

    # Walk through the predictions and greedily match to the first available ground truth
    # Some predictions will not be matched
    greedy_matches = []
    assigned_gts = []
    for pred in preds:
        for gt in pred2gt[pred]:
            if gt not in assigned_gts:
                greedy_matches.append((pred, gt))
                assigned_gts.append(gt)
                break
    return greedy_matches

def _get_num_swaps(greedy_matches: List[Tuple[int, int]]) -> int:
    """
    Get the number of swaps in the greedy matches.
    """
    num_swaps = 0
    for i in range(len(greedy_matches)):
        for j in range(i + 1, len(greedy_matches)):
            if greedy_matches[i][1] > greedy_matches[j][1]:
                num_swaps += 1
    return num_swaps

def _calculate_precision(n_pred: int, n_gt: int, greedy_matches: List[Tuple[int, int]]) -> float:
    if n_pred == 0:
        return 0.0
    return len(greedy_matches) / n_pred

def _calculate_recall(n_pred: int, n_gt: int, greedy_matches: List[Tuple[int, int]]) -> float:
    if n_gt == 0:
        return 0.0
    return len(greedy_matches) / n_gt

def _calculate_f1(n_pred: int, n_gt: int, greedy_matches: List[Tuple[int, int]]) -> float:
    precision = _calculate_precision(n_pred, n_gt, greedy_matches)
    recall = _calculate_recall(n_pred, n_gt, greedy_matches)
    if (precision + recall) == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)

def _calculate_ordering_score(n_pred: int, n_gt: int, greedy_matches: List[Tuple[int, int]]) -> float:
    num_swaps = _get_num_swaps(greedy_matches)
    num_matches = len(greedy_matches)
    if num_matches <= 1:
        return 1.0
    return 1.0 - num_swaps / (num_matches * (num_matches - 1) / 2)

def _calculate_summary_steps_score(
    n_pred: int,
    n_gt: int,
    greedy_matches: List[Tuple[int, int]]
) -> float:
    f1 = _calculate_f1(n_pred, n_gt, greedy_matches)
    ordering = _calculate_ordering_score(n_pred, n_gt, greedy_matches)
    return 100. * f1 * ordering

def _get_scores(pred_steps: List[str], gt_steps: List[str]) -> dict[str, float]:
    """Get scores for the predicted steps against the ground truth steps."""
    semantic_matches = _get_bipartite_matching(pred_steps, gt_steps)
    greedy_matches = _get_greedy_matches(semantic_matches)
    n_pred = len(pred_steps)
    n_gt = len(gt_steps)

    greedy_matches_1indexed = [(x[0] + 1, x[1] + 1) for x in greedy_matches]

    return {
        **{
            metric: eval(f"_calculate_{metric}")(n_pred, n_gt, greedy_matches) \
                for metric in SUMMARY_STEPS_METRICS
        },
        "pred2gt_matches": str(greedy_matches_1indexed)
    }

def sr_summary_doc_to_visual(doc):
    return [VIDEO_DIR + doc["path_v"]]

def sr_summary_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = (
        "List the actions performed by the patient in the video.\n\n" \
        "Example Response:\n" \
        "```\n"
        "- remove glasses\n" \
        "- wring the washcloth\n" \
        "- twist the deodorant cap\n" \
        "- place cup on table\n" \
        "- brush teeth\n" \
        "```\n" \
        "Often, the list may only be one to three action items long, but can up to 8 items long.\n"
    )
    return prompt

def sr_summary_doc_to_target(doc):
    activity = doc["activity"]
    gt_sequence = activity2steps.get(activity, [])
    if not gt_sequence:  # hopefully, doesn't happen?
        raise ValueError("No steps found for activity: ", activity)
    return gt_sequence

def sr_summary_process_results(doc, results):
    """Process per-document results into metric format"""
    gt_steps = sr_summary_doc_to_target(doc)
    pred_steps = _parse_summary_results(results)
    scores = _get_scores(pred_steps, gt_steps)
    return {
        **doc,
        "pred": "[" + ",".join(pred_steps) + "]",
        **scores
    }
