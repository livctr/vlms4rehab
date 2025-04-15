import gc
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
    if qwen2_llm_judge is None:
        torch.cuda.empty_cache()
        gc.collect()
        qwen2_llm_judge = pipeline(
            task="text-generation",
            model="Qwen/Qwen2-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
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


def sr_describe_doc_to_visual(doc):
    return [VIDEO_DIR + doc["path_v"]]

def sr_describe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = (
        "Describe what the patient does in the video."
    )
    return prompt

def sr_describe_doc_to_target(doc):
    activity = doc["activity"]
    gt_sequence = activity2steps.get(activity, [])
    if not gt_sequence:  # hopefully, doesn't happen?
        raise ValueError("No steps found for activity: ", activity)
    return gt_sequence

def sr_describe_process_results(doc, results):
    """Process per-document results into metric format"""
    global qwen2_llm_judge
    if qwen2_llm_judge is None:
        _init_llm_judge()

    gt_steps = sr_describe_doc_to_target(doc)
    predicted_actions = results[0]

    gt_steps_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(gt_steps)])
    prompt = (
        f"You are an impartial judge given the activity and instructions a patient follows in a video. "
        f"Your task is to rate the quality of a generated description of the patient's actions on a scale from 0 to 100, "
        f"0 being the worst and 100 being the best. \n"
        f"Activity: {doc['activity']}\n"
        f"Instructions:\n{gt_steps_str}\n"
        f"Generated description:\n{predicted_actions}\n\n"
        f"Give your thought process first. Then, assign your final score formatted as 'Final score: <score>'. "
    )
    completion = _get_completion(
        prompt=prompt,
        max_new_tokens=512,
    )
    eval_logger.debug(f"Completion: {completion['content']}")

    # Extract the score from the completion
    matches = re.findall(r"[Ss]core: (\d+)", completion["content"])
    if len(matches) > 0:
        score = int(matches[0])
    else:
        matches = re.findall(r'\d+', completion["content"])
        if len(matches) > 0:
            score = int(matches[-1])
        else:
            eval_logger.debug(f"No score found in the completion: {completion['content']}")
            score = 0

    return {
        **doc,
        "pred": predicted_actions,
        "evaluation": completion["content"],
        "llm_score": score,
    }
