from loguru import logger as eval_logger
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from datasets import Dataset, DatasetDict
import os
import datasets
import pandas as pd
import yaml
import glob
import re
from functools import partial
from typing import List, Tuple
from openai import OpenAI


SUMMARY_STEPS_METRICS = ["precision", "recall", "f1", "ordering_score", "summary_steps_score"]

client = OpenAI()
cost = 0.0
model = "gpt-4o-mini"
model_input_cost_per_mil = 0.15
model_output_cost_per_mil = 0.60
cost_log_freq = 0.0001  # logs cost every 0.01 dollars


METADATA_PATH = "./data/csvs_txts_yamls/cleaned_metadata.csv"
HEALTHY_PATIENTS = (
    "C00011,C00012,C00015,C00019,C00020,C00022,C00023,C00024,C00025,C00026,"
    "C00027,C00028,C00029,C00030,C00031,C00032,C0004,C0005,C0007,C0009"
)
MILD_PATIENTS = (
    "S0005,S0007,S0009,S00010,S00012,S00013,S00016,S00023,S00026,S00028,"
    "S00030,S00032,S00033,S00035,S00037,S00040,S00041,S00042,S00043,S00047"
)
MODERATE_PATIENTS = (
    "S0001,S0002,S0003,S0004,S0006,S0008,S00011,S00017,S00018,S00019,S00020,"
    "S00022,S00024,S00025,S00027,S00036,S00039,S00044,S00045,S00046,S00048,"
    "S00053,S00054"
)
SEVERE_PATIENTS = (
    "S00021,S00029,S00031,S00034,S00049,S00050,S00051,S00055"
)
PATIENTS = (
    HEALTHY_PATIENTS + "," + \
    MILD_PATIENTS + "," + \
    MODERATE_PATIENTS + "," + \
    SEVERE_PATIENTS
)
ACTIVITY_GROUND_TRUTH_PATH = "./data/csvs_txts_yamls/activities_ground_truth.yaml"
try:
    with open(ACTIVITY_GROUND_TRUTH_PATH, "r") as file:
        agt = yaml.safe_load(file)
    activity2steps = {x.get("name"): x.get("steps") for x in agt}
except Exception as e:
    print(f"An error occurred while reading the YAML file: {e}")
    raise e


def strokerehab_load_dataset(patients='all', activity='all', reps='all', filter_for_testset=False):
    """
    Loads the StrokeRehab dataset from a cleaned metadata file and applies AND-ed filters.

    Args:
        patients (str): The patient IDs to include in the dataset. Default is 'all'.
            If specifying individual patients, separate them with commas
            Example: 'S0001,S0002'
        activity (str): The activity names to include in the dataset. Default is 'all' (11 total).
            If specifying individual activities, separate them with commas
            Example: 'RTT left side,RTT right side,brushing,combing,deodrant,drinking,face wash,feeding,glasses,shelf left side,shelf right side'
        reps (str): Either 'all' or 'first'.
        filter_for_testset (bool): If True, include only the VIDEOS in the test set of
            the original StrokeRehab paper. https://pubmed.ncbi.nlm.nih.gov/37766938/
            This data is located in './data/csvs_txts_yamls/strokerehab_test_set.txt' and already
            incorporated into the CSV metadata file.
    
    Returns:
        dataset (datasets.Dataset): The StrokeRehab dataset with the specified filters applied.

    """
    df = pd.read_csv(METADATA_PATH)
    if patients != 'all':
        patients = patients.split(',')
        df = df[df['patient'].isin(patients)]
    if activity != 'all':
        activity = activity.split(',')
        df = df[df['activity'].isin(activity)]
    if reps != 'all':
        if reps != 'first':
            raise ValueError("Invalid value for reps. Must be 'all' or 'first'.")
        df = df.sort_values('id').groupby(['patient', 'activity']).agg('first').reset_index()
    if filter_for_testset:
        df = df[df['is_in_strokerehab_test_set']]
    dataset = datasets.Dataset.from_pandas(df)
    dataset_dict = datasets.DatasetDict({'test': dataset})
    return dataset_dict

strokerehab_load_dataset_debug = partial(strokerehab_load_dataset, patients='S0001', activity='face wash,glasses', reps='first')
strokerehab_load_dataset_S0001 = partial(strokerehab_load_dataset, patients='S0001')
strokerehab_load_dataset_3patients = partial(strokerehab_load_dataset, patients='C00011,S0001,S0002')
strokerehab_load_dataset_onerep = partial(strokerehab_load_dataset, reps='first')
strokerehab_load_dataset_test = partial(strokerehab_load_dataset, filter_for_testset=True)
strokerehab_load_dataset_healthy = partial(strokerehab_load_dataset, patients=HEALTHY_PATIENTS)
strokerehab_load_dataset_mild = partial(strokerehab_load_dataset, patients=MILD_PATIENTS)
strokerehab_load_dataset_moderate = partial(strokerehab_load_dataset, patients=MODERATE_PATIENTS)
strokerehab_load_dataset_severe = partial(strokerehab_load_dataset, patients=SEVERE_PATIENTS)


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


def parse_summary_results(results: str):
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
    global cost

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
        prev_cost = cost
        cost += (response["input_tokens"] * model_input_cost_per_mil +
                    response["output_tokens"] * model_output_cost_per_mil) / 1e6
        if int(prev_cost // cost_log_freq) != int(cost // cost_log_freq):
            eval_logger.info(f"{model}-as-a-judge cost: ${cost:.4f}")

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
    return f1 * ordering

def get_scores(pred_steps: List[str], gt_steps: List[str]) -> dict[str, float]:
    """Get scores for the predicted steps against the ground truth steps."""
    semantic_matches = _get_bipartite_matching(pred_steps, gt_steps)
    greedy_matches = _get_greedy_matches(semantic_matches)
    n_pred = len(pred_steps)
    n_gt = len(gt_steps)

    return {
        metric: eval(f"_calculate_{metric}")(n_pred, n_gt, greedy_matches) for metric in SUMMARY_STEPS_METRICS
    }

def sr_summary_doc_to_visual(doc):
    return [doc["path_v"]]

def sr_summary_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = "Outline the steps the patient goes through to accomplish their task."
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
    pred_steps = parse_summary_results(results[0])
    scores = get_scores(pred_steps, gt_steps)
    return {
        **doc,
        "pred": "[" + ",".join(pred_steps) + "]",
        **scores
    }


################################# METRICS ####################################

def calculate_edit_similarity(pred, ref):
    """Normalized sequence similarity using Levenshtein distance"""
    if not pred and not ref:
        return 1.0  # Both empty sequences
    max_len = max(len(pred), len(ref))
    edit_dist = levenshtein_distance(pred, ref)
    return 1 - (edit_dist / max_len)

def calculate_precision(pred, ref):
    """Action-level Precision score"""
    common = set(pred) & set(ref)
    precision = len(common) / len(pred) if pred else 0
    return precision

def calculate_recall(pred, ref):
    """Action-level Recall score"""
    common = set(pred) & set(ref)
    recall = len(common) / len(ref) if ref else 0
    return recall

def calculate_action_f1(pred, ref):
    """Action-level F1 score (order-agnostic)"""
    common = set(pred) & set(ref)
    precision = len(common) / len(pred) if pred else 0
    recall = len(common) / len(ref) if ref else 0
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
