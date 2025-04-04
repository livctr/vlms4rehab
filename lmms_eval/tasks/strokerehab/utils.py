import os
import glob
from loguru import logger as eval_logger
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import Dataset, DatasetDict
import os
import datasets
import re
import glob
import json

from lmms_eval.api.task import ConfigurableTask

dataset_path = r"/gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/naveen/test_data"

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


ACTION_LABELS = sorted(['Reach', 'Reposition', 'Transport', 'Stabilization', 'Idle'])
ACTIVITY_DESCS = {
    "face wash": {
        "Activity": "Face Washing",
        "Target": "washcloths, faucet handle, tub",
        "Instructions": "Fill tub with water. Dip washcloth on the right side into water. Wring the washcloth. Wipe each side of face with wet washcloth. Place first washcloth back on countertop. Use washcloth on the left side to dry face. Place second washcloth back on countertop.",
        "Workspace": "Sink with a small tub (32.3 x 24.1 x 2.5 cm^3) in it and two folded wash-cloths on either side of the counter-top, 30 cm from edge closest to patient"
    },

    "deodrant": {
        "Activity": "Applying Deodorant",
        "Target": "deodorant (solid twist-base)",
        "Instructions": "Remove deodorant cap. Twist deodorant base a few times. Apply deodorant. Replace deodorant cap. Untwist the deodorant base. Put deodorant on table.",
        "Workspace": "Tabletop with deodorant placed at midline, 25 cm from edge closest to patient"
    },

    "combing": {
        "Activity": "Hair Combing",
        "Target": "comb",
        "Instructions": "Pick up comb. Comb both sides of head.",
        "Workspace": "Tabletop with comb placed at midline, 25 cm from edge closest to patient"
    },

    "glasses": {
        "Activity": "Putting On and Taking Off Glasses",
        "Target": "pair of glasses",
        "Instructions": "Wear glasses. Return hands to table. Remove glasses. Place glasses on table.",
        "Workspace": "Tabletop with glasses placed at midline, 25 cm from edge closest to patient"
    },

    "feeding": {
        "Activity": "Feeding",
        "Target": "paper plate, fork, knife, re-sealable sandwich baggie, slice of bread, single-serve margarine container",
        "Instructions": "Remove bread from plastic bag. Put the bread on plate. Open margarine pack. Spread margarine on bread. Cut bread into four pieces. Cut off some bread. Eat a small bite-sized piece of bread.",
        "Workspace": "Tabletop with a standard-size paper plate (21.6 cm diameter) placed at midline, 2 cm from edge, utensils placed 3 cm from edge, 5 cm from either side of plate, a baggie with a slice of bread placed 25 cm from edge, 23 cm left of midline, and a margarine packet placed 32 cm from edge, 17 cm right of midline"
    },

    "drinking": {
        "Activity": "Drinking Water",
        "Target": "water bottle (12 oz), paper cup (4 oz)",
        "Instructions": "Open water bottle. Pour water into cup. Take a sip of water. Place cup on table. Replace cap on bottle.",
        "Workspace": "Tabletop with water bottle and paper cup 18 cm to the left and right of midline, 25 cm from edge closest to patient"
    },

    "brushing": {
        "Activity": "Teeth Brushing",
        "Target": "travel-sized toothpaste, toothbrush with built-up foam grip, faucet handle",
        "Instructions": "Wet toothbrush. Apply toothpaste to toothbrush. Replace cap on toothpaste tube. Brush teeth. Rinse toothbrush and mouth. Place toothbrush back on countertop.",
        "Workspace": "Sink with toothpaste and toothbrush on either side of the countertop, 30 cm from edge closest to patient"
    },

    "RTT left side": {
        "Activity": "Repetitive Target Touching (Left Side)",
        "Target": "toilet paper roll wrapped in self-adhesive wrap",
        "Instructions": "Move toilet paper roll between center and outside target repeatedly.",
        "Workspace": "Horizontal circular array (48.5 cm diameter) of 8 targets (5 cm diameter)"
    },

    "RTT right side": {
        "Activity": "Repetitive Target Touching (Right Side)",
        "Target": "toilet paper roll wrapped in self-adhesive wrap",
        "Instructions": "Move toilet paper roll between center and outside target repeatedly.",
        "Workspace": "Horizontal circular array (48.5 cm diameter) of 8 targets (5 cm diameter)"
    },

    "shelf left side": {
        "Activity": "Shelf Reach and Place (Left Side)",
        "Target": "toilet paper roll wrapped in self-adhesive wrap",
        "Instructions": "Move toilet paper roll between center target and various shelf targets repeatedly.",
        "Workspace": "Shelf with two levels (33 cm and 53 cm) with 3 targets on both levels (22.5 cm, 45 cm, and 67.5 cm away from the left-most edge)"
    },

    "shelf right side": {
        "Activity": "Shelf Reach and Place (Right Side)",
        "Target": "toilet paper roll wrapped in self-adhesive wrap",
        "Instructions": "Move toilet paper roll between center target and various shelf targets repeatedly.",
        "Workspace": "Shelf with two levels (33 cm and 53 cm) with 3 targets on both levels (22.5 cm, 45 cm, and 67.5 cm away from the left-most edge)"
    }
}


def build_breakfast_dataset():
    print("StrokeRehab Dataset Activated in utils.py")
    examples = []

    video_dir = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM"
    label_dir = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels"

    video_dataset = []
    label_dataset = []
    video_id_dataset = []
    activity_dataset = []
    csv_file = os.path.join(dataset_path, "cleaned_metadata.csv")
    with open(csv_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:25]:
            video_id_dataset.append(line.split(",")[0])
            video_dataset.append(os.path.join(video_dir, line.split(",")[2]))
            activity_dataset.append(line.split(",")[5])
            label_dataset.append(os.path.join(label_dir, line.split(",")[10]))

    #for video_path in glob.glob(os.path.join(dataset_path, "**/*.avi"), recursive=True)[:5]:
    for idx in range(len(video_dataset)):
        label_path = label_dataset[idx]
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                first_line = f.readline().split(",")
                marker_idx = first_line.index("MarkerNames\n")
                seq = [line.strip().split(",")[marker_idx] for line in f]

            curr_label = None
            final_seq = []
            for elem in seq[1:]:
                if curr_label != elem:
                    final_seq.append(elem)
                    curr_label = elem
            
            primitives_seq = []
            for elem in final_seq:
                if "idle" in elem.lower():
                    primitives_seq.append("Idle")
                elif "reach" in elem.lower():
                    primitives_seq.append("Reach")
                elif "reposition" in elem.lower():
                    primitives_seq.append("Reposition")
                elif "transport" in elem.lower():
                    primitives_seq.append("Transport")
                elif "stabilization" in elem.lower():
                    primitives_seq.append("Stabilization")

            curr_label = None
            final_seq = []
            for elem in primitives_seq:
                if curr_label != elem:
                    final_seq.append(elem)
                    curr_label = elem

            examples.append({
                "video_path": video_dataset[idx],
                "ground_truth": final_seq,
                "video_id": video_id_dataset[idx],
                "activity": activity_dataset[idx]
            })

    print("DATASETS: ")
    print(datasets.DatasetDict({
        "test": datasets.Dataset.from_dict({
            "video_path": [ex["video_path"] for ex in examples],
            "ground_truth": [ex["ground_truth"] for ex in examples],
            "video_id": [ex["video_id"] for ex in examples],
            "activity": [ex["activity"] for ex in examples]
        })
    }))

    return datasets.DatasetDict({
        "test": datasets.Dataset.from_dict({
            "video_path": [ex["video_path"] for ex in examples],
            "ground_truth": [ex["ground_truth"] for ex in examples],
            "video_id": [ex["video_id"] for ex in examples],
            "activity": [ex["activity"] for ex in examples]
        })
    })

def breakfast_doc_to_visual(doc):
    return [doc["video_path"]]

def breakfast_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    functional_primitives = {
        "Reach": "Arm/hand movement to touch or grasp a target object",
        "Reposition": "Arm/hand movement to adjust position near a target object without moving the object itself",
        "Transport": "Arm/hand movement to carry or move a held object from one location to another",
        "Stabilization": "Arm/hand holds an object still with minimal motion",
        "Idle": "Arm/hand is inactive but positioned near an object, ready to act"
    }
    
    # Create a formatted list of primitives
    primitives_list = "\n".join([f"- {primitive}: {description}" for primitive, description in functional_primitives.items()])
    
    prompt = f"""Analyze the video of the patient's activity thoroughly. Let me give you some context for the video.

    **Activity:** {ACTIVITY_DESCS[doc["activity"]]["Activity"]}
    **Workspace:** {ACTIVITY_DESCS[doc["activity"]]["Workspace"]}
    **Target Object:** {ACTIVITY_DESCS[doc["activity"]]["Target"]}
    **Instructions:** {ACTIVITY_DESCS[doc["activity"]]["Instructions"]}

    Your task is to identify the sequence of highly granular activitives performed by the hand. Describe the hand movements chronologically.
    Then, map each action to ONE of the predefined action labels below. Only use these exact labels - do not create new ones.

    List of valid action labels with descriptions:
    {primitives_list}


    **Output Format:**

        PART 1: ACTION DESCRIPTION
        1. Action 1
        2. Action 2
        3. Action 3
        .
        .
        .
        N. Action N

    PART 2: ACTION LABELS
    1. Action Label 1
    2. Action Label 2
    3. Action Label 3
    .
    .
    .
    N. Action Label N
    
    Make sure each action label exactly matches one from the provided list.

    """


    return prompt


def process_generation(doc, text):
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

    # Remove duplicates while preserving order
    curr_act = None
    final_act_set = []
    for elem in detected_actions:
        if curr_act != elem:
            final_act_set.append(elem)
            curr_act = elem

    print("DETECTED ACTIONS: ", final_act_set)
    print("GROUND TRUTH: ", doc["ground_truth"])

    result = {
        "gt_answer": doc["ground_truth"],
        "answer_prediction": final_act_set,
    }

    return final_act_set


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
        category = video_id.split('_')[0]
        
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