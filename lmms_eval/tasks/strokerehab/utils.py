import os
import glob
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


METRICS = ["edit_similarity", "precision", "recall", "action_f1"]



# Patients in dataset
# C00011,C00012,C00015,C00019,C00020,C00022,C00023,C00024,C00025,C00026,C00027
# C00028,C00029,C00030,C00031,C00032,C0004,C0005,C0007,C0009,S0001,S00010
# S00011,S00012,S00013,S00016,S00017,S00018,S00019,S0002,S00020,S00021,S00022
# S00023,S00024,S00025,S00026,S00027,S00028,S00029,S0003,S00030,S00031,S00032
# S00033,S00034,S00035,S00036,S00037,S00039,S0004,S00040,S00041,S00042,S00043
# S00044,S00045,S00046,S00047,S00048,S00049,S0005,S00050,S00051,S00053,S00054
# S00055,S0006,S0007,S0008,S0009

METADATA_PATH = "./data/csvs_txts_yamls/cleaned_metadata.csv"
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
    df = pd.read_csv("./data/csvs_txts_yamls/cleaned_metadata.csv")
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

strokerehab_load_dataset_S0001 = partial(strokerehab_load_dataset, patients='S0001')


def strokerehab_doc_to_visual(doc):
    return [doc["path_v"]]

def strokerehab_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = "Summarize this video!"  # TODO prompt engineer
    return prompt

def strokerehab_doc_to_target(doc):
    activity = doc["activity"]
    gt_sequence = activity2steps.get(activity, [])
    if not gt_sequence:  # hopefully, doesn't happen?
        raise ValueError("No steps found for activity: ", activity)
    return gt_sequence

def strokerehab_process_results(doc, results):
    """Process per-document results into metric format"""
    gt_sequence = strokerehab_doc_to_target(doc)
    pred_sequence = [results[0], "remove glasses"]  # TODO: Replace with actual prediction

    scores = {metric: eval("calculate_" + metric)(pred_sequence, gt_sequence) for metric in METRICS}

    # model?

    return {
        "sr_summary_score": {
            **doc,
            "pred": "[" + ",".join(pred_sequence) + "]",
            **scores
        }
    }

def strokerehab_aggregate_results(results):
    """Calculate final scores across all samples"""
    results = pd.DataFrame(results)
    results.to_csv("results.csv", index=False)
    print("RESULTS")
    print('---------------------------------------')
    print(results.head())
    print('---------------------------------------')

    overall_metrics = {}
    for metric in METRICS:
        overall_metrics[metric] = results[metric].mean()
    return overall_metrics


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
