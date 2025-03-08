import os
import glob
from loguru import logger as eval_logger
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import Dataset, DatasetDict
import os
import datasets
import glob

from lmms_eval.api.task import ConfigurableTask

dataset_path = r"C:\Users\AI Research\Desktop\work\BreakfastII_15fps_qvga_sync\BreakfastII_15fps_qvga_sync"

class BreakfastTask(ConfigurableTask):
    def __init__(self, config, model_name):
        super().__init__(config, model_name)
        
    def has_test_docs(self):
        return True
        
    def test_docs(self):
        return self.dataset["test"]
        
    def doc_to_text(self, doc):
        return breakfast_doc_to_text(doc)
        
    def doc_to_visual(self, doc):
        return breakfast_doc_to_visual(doc)
        
    def doc_to_target(self, doc):
        return [doc["ground_truth"]]
        
    def process_results(self, doc, results):
        return breakfast_process_results(doc, results)

ACTION_LABELS = sorted(['SIL', 'take_bowl', 'pour_cereals', 'pour_milk', 'stir_cereals', 'take_cup', 'pour_coffee', 'pour_oil', 'crack_egg', 'fry_egg', 'put_egg2plate', 'spoon_powder',
 'stir_milk', 'take_knife', 'cut_fruit', 'put_fruit2bowl', 'peel_fruit', 'stir_fruit', 'cut_bun', 'take_butter', 'smear_butter', 'take_topping', 'put_toppingOnTop',
 'add_teabag', 'pour_water', 'walk_in', 'walk_out', 'take_plate', 'spoon_flour', 'stir_dough', 'pour_dough2pan', 'fry_pancake', 'put_pancake2plate', 'put_bunTogether', 'stir_egg', 'pour_egg2pan',
 'stirfry_egg', 'add_saltnpepper', 'pour_sugar', 'cut_orange', 'take_squeezer', 'squeeze_orange', 'pour_juice', 'take_eggs', 'take_glass', 'butter_pan', 'pour_flour',
 'spoon_sugar', 'stir_coffee', 'stir_tea'])


def build_breakfast_dataset():
    print("Breakfast Dataset Activated in utils.py")
    examples = []
    # Choosing 10 subset values only 
    for video_path in glob.glob(os.path.join(dataset_path, "**/*.avi"), recursive=True)[:5]:
        label_path = video_path + ".labels"
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                seq = [line.strip().split()[1] for line in f]
            
            examples.append({
                "video_path": video_path,
                "ground_truth": seq,
                "video_id": os.path.basename(video_path).split('.')[0]
            })

    return datasets.DatasetDict({
        "test": datasets.Dataset.from_dict({
            "video_path": [ex["video_path"] for ex in examples],
            "ground_truth": [ex["ground_truth"] for ex in examples],
            "video_id": [ex["video_id"] for ex in examples]
        })
    })

def breakfast_doc_to_visual(doc):
    return [doc["video_path"]]

def breakfast_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = (
        "Analyze this video and list actions happening in the VIDEO using these labels:\n"
        f"{', '.join(ACTION_LABELS)}\n\n"
        "Format your response as:\n"
        "1. [First Action]\n"
        "2. [Next Action]\n"
        "...\n"
        "N. [Final Action]\n\n"
    )
    return prompt

def process_generation_legacy(text):
    # Extract action sequence from model output
    detected_actions = []
    for action in ACTION_LABELS:
        if action in text.lower():
            detected_actions.append(action)
    return detected_actions

import re

def process_generation(doc, text):
    # Extract ordered action sequence from model output 
    # Find all numbered items
    numbered_items = re.findall(r'\d+\.\s*(.+?)\s*(?=\d+\.|$)', text, flags=re.DOTALL)
    
    # Find bulleted items if numbered not found
    if not numbered_items:
        numbered_items = re.findall(r'-\s*(.+?)\s*(?=-\s|$)', text, flags=re.DOTALL)
    
    detected_actions = []
    normalized_labels = {label.lower().replace('_', ' '): label for label in ACTION_LABELS}
    
    for item in numbered_items:
        clean_item = item.strip().lower().replace('_', ' ').replace('-', ' ')
        
        # Find best match from action labels
        for label_phrase, original_label in normalized_labels.items():
            if re.search(r'\b' + re.escape(label_phrase) + r'\b', clean_item):
                detected_actions.append(original_label)
                break 
    
    print(" DETECTED ACTIONS: ", detected_actions)
    print("GROUND TRUTH: ", doc["ground_truth"])

    result = {
        "gt_answer": doc["ground_truth"],
        "answer_prediction": detected_actions,
    }

    return detected_actions


def breakfast_process_results(doc, results):
    """Process per-document results into metric format"""
    print(results)
    print("TEST \n")
    print(results[0])
    pred_sequence = process_generation(doc, results[0])
    gt_sequence = doc["ground_truth"]
    
    return {
        "breakfast_score": {
            "video_id": doc["video_id"],
            "precision": calculate_precision(pred_sequence, gt_sequence),
            "recall": calculate_recall(pred_sequence, gt_sequence),
            "edit_similarity": calculate_edit_distance(pred_sequence, gt_sequence),
            "action_f1": calculate_f1(pred_sequence, gt_sequence)
        }
    }

def breakfast_aggregate_results(results):
    """Calculate final scores across all samples"""
    category_scores = defaultdict(list)
    total_scores = defaultdict(float)
    
    # Collect scores by video category (if available)
    for result in results:
        video_id = result["video_id"]
        category = video_id.split('_')[0]  # Example: 'P03' in 'P03_cereals'
        
        category_scores[category].append({
            "precision": result["precision"],
            "recall": result["recall"],
            "edit_similarity": result["edit_similarity"],
            "action_f1": result["action_f1"]
        })
        
        for metric in ["precision", "recall", "edit_similarity", "action_f1"]:
            total_scores[metric] += result[metric]

    # Calculate category averages
    category_metrics = {}
    for category, scores in category_scores.items():
        category_metrics[category] = {
            metric: sum(s[metric] for s in scores) / len(scores)
            for metric in ["precision", "recall", "edit_similarity", "action_f1"]
        }

    # Calculate overall averages
    num_samples = len(results)
    overall_metrics = {
        metric: total_scores[metric] / num_samples
        for metric in ["precision", "recall", "edit_similarity", "action_f1"]
    }

    # Print formatted results
    eval_logger.info("Breakfast Action Recognition Metrics:")
    eval_logger.info("Category Breakdown:")
    for category, metrics in category_metrics.items():
        eval_logger.info(f"  {category}:")
        eval_logger.info(f"    Precision: {metrics['precision']:.2%}")
        eval_logger.info(f"    Recall: {metrics['recall']:.2%}")
        eval_logger.info(f"    Edit Similarity: {metrics['edit_similarity']:.2%}")
        eval_logger.info(f"    Action F1: {metrics['action_f1']:.2%}")

    eval_logger.info("\nOverall Scores:")
    eval_logger.info(f"Precision: {overall_metrics['precision']:.2%}")
    eval_logger.info(f"Recall: {overall_metrics['recall']:.2%}")
    eval_logger.info(f"Edit Similarity: {overall_metrics['edit_similarity']:.2%}")
    eval_logger.info(f"Action F1: {overall_metrics['action_f1']:.2%}")

    return overall_metrics


def calculate_edit_distance(pred, ref):
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


def calculate_f1(pred, ref):
    """Action-level F1 score (order-agnostic)"""
    common = set(pred) & set(ref)
    precision = len(common) / len(pred) if pred else 0
    recall = len(common) / len(ref) if ref else 0
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)