from typing import List, Tuple

from Levenshtein import distance as levenshtein_distance
from loguru import logger as eval_logger
from openai import OpenAI
import re
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
try:
    client = OpenAI()
    cost = 0.0
    model = "gpt-4o-mini"
    model_input_cost_per_mil = 0.15
    model_output_cost_per_mil = 0.60
    cost_log_freq = 0.0001  # logs cost every 0.01 dollars

    def _log_cost(input_tokens, output_tokens):
        global cost
        prev_cost = cost
        cost += (input_tokens * model_input_cost_per_mil + \
                 output_tokens * model_output_cost_per_mil) / 1e6
        if int(prev_cost // cost_log_freq) != int(cost // cost_log_freq):
            eval_logger.info(f"{model}-as-a-judge cost: ${cost:.4f}")

except Exception as e:
    print(f"Need API key for OpenAI to run this code. Error: {e}")


def _get_completion(client: OpenAI, 
                    prompt: str, 
                    model: str = "gpt-4o-mini", 
                    max_tokens: int = 32,
                    temperature: float = 0.0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return {
        "content": response.choices[0].message.content.strip(),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }


def _parse_summary_results(results: str):
    """
    Expects a numbered list of steps, potentially not separated by newlines.

    Example input:
    "
    1. remove glasses
    2. wash hands
    3. apply soap
    "
    Parses this into a list of strings. Throws error if format is incorrect.

    Example output: ["remove glasses", "wash hands", "apply soap"]
    """
    # Remove any leading/trailing whitespace from the input.
    results = results.strip()
    
    # Split the string by a pattern matching one or more digits followed by a dot.
    parts = re.split(r'\d+\.', results)

    # Remove any empty strings and extra whitespace from each step.
    steps = []
    for step in parts:
        step = step.strip()
        if step:
            steps.append(step)
    return steps


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
        response = _get_completion(client, prompt, model=model)

        # Log cost
        _log_cost(response["input_tokens"], response["output_tokens"])

        # Create bipartite match graph
        if pred_step_str in cache:
            matches_deduped_idx = cache[pred_step_str]
        else:
            matches_deduped_idx = []
            if response["content"] != "None":
                try:
                    for x in response["content"].split(","):
                        x = int(x) - 1
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
        "Provide a numbered step-by-step breakdown of what the patient does in the video. "
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
    pred_steps = _parse_summary_results(results[0])
    scores = _get_scores(pred_steps, gt_steps)
    return {
        **doc,
        "pred": "[" + ",".join(pred_steps) + "]",
        **scores
    }
