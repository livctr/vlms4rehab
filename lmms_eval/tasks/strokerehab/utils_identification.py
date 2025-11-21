
from functools import partial
import os
import re

from data.utils_strokerehab import DataPaths
from lmms_eval.tasks.strokerehab.utils_primitives import load_strokerehab_primitives_dataset

NORMAL_PROMPT = """
{
  "question": "Which activity is the patient performing in this video?",
  "response_instructions": "After noting your observations, end your reply with exactly one line: FINAL_ANSWER: <activity_name>",
  "instructions": "Determine the main activity being performed by the patient. Choose the most fitting activity label from the list below. Always respond with an activity, even if uncertain.",
  "activity_classes": {
    "Brushing": "The patient applies toothpaste to a toothbrush, brushes their teeth, rinses, and sets the brush back down.",
    "Combing": "The patient picks up a comb and combs both sides of their hair.",
    "Deodorant": "The patient twists open a deodorant stick, applies it under the arm, then replaces the cap.",
    "Drinking": "The patient pours water from a bottle into a cup, takes a sip, and replaces the cap.",
    "Face wash": "The patient washes and dries their face using two washcloths at a sink.",
    "Feeding": "The patient prepares bread with margarine on a plate and eats a small piece using utensils.",
    "Glasses": "The patient puts on or removes a pair of glasses from the tabletop.",
    "RTT exercise": "The patient slides a toilet paper roll between center and outer targets on a flat surface.",
    "Shelf exercise": "The patient transfers a toilet paper roll between the center target and multiple shelf levels."
  }
}
"""


TUNED_PROMPT = """
{
  "question": "Which activity is the patient performing in this video?",
  "response_instructions": "After noting your observations, end your reply with exactly one line: FINAL_ANSWER: <activity_name>",
  "instructions": "Determine the main activity being performed by the patient. Choose the most fitting activity label from the list below. Always respond with an activity, even if uncertain.",
  "activity_classes": {
    "Brushing": "The patient is at the sink. Either toothpaste or a toothbrush is visible, indicating the patient is brushing their teeth.",
    "Combing": "The patient grabs a small rectangular object (likely a comb) on the table and moves it near the hair area (likely to groom their hair).",
    "Deodorant": "The patient applies deodorant using a deodorant tube (likely white) on the table. The hand is seen moving towards the underarm area.",
    "Drinking": "The patient pours water from a plastic water bottle into a cylindrical cup on the table and takes a sip.",
    "Face wash": "The patient washes their face at the sink using water and a wash cloth or towel.",
    "Feeding": "The patient prepares and eats bread on a white plate by spreading margarine, cutting it, and taking bites.",
    "Glasses": "The patient grabs a pair of glasses from the table and puts them on or removes them.",
    "RTT exercise": "The patient repeatedly moves a cylindrical block on the table. ",
    "Shelf exercise": "The patient repeatedly moves a cylindrical block on a TRANSPARENT shelf.",
  }
}
"""


FINAL_ANSWER_RE = re.compile(
    r'(?mi)FINAL_ANSWER\s*:\s*([^\n<]+)',  # capture until a '<' or newline
    re.MULTILINE,
)

def _parse_final_answer(text: str, allowed_labels: set[str] | None = None) -> str:
    """
    Extract the activity name from a model's output line:
        FINAL_ANSWER: <activity_name>

    - Case-insensitive match for the tag.
    - Ignores leading/trailing whitespace.
    - Strips wrapping quotes/backticks and benign trailing punctuation.
    - If `allowed_labels` is provided, returns "" when the extracted label isn't recognized.
    - Returns "" if no FINAL_ANSWER line is found.
    """
    if not text:
        return ""

    m = FINAL_ANSWER_RE.search(text)
    if not m:
        return ""

    val = m.group(1).strip()

    # Strip surrounding quotes/backticks if present
    if len(val) >= 2:
        pairs = {('"', '"'), ("'", "'"), ('`', '`'), ('“', '”'), ('‘', '’')}
        for lq, rq in pairs:
            if val.startswith(lq) and val.endswith(rq):
                val = val[1:-1].strip()
                break

    # Trim common trailing punctuation that models sometimes add
    val = val.rstrip(" .;:,`'\"")

    # Optional: enforce whitelist
    if allowed_labels is not None and val not in allowed_labels:
        return ""

    return val


def sr_id_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [os.path.join(DataPaths.RAW_VIDEO_DIR, doc["path_v"])]

def sr_id_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is not None and lmms_eval_specific_kwargs.get("prompt", "normal") == "tuned":
        prompt = TUNED_PROMPT
    else:
        prompt = NORMAL_PROMPT
    return prompt


def sr_id_doc_to_target(doc):
    activity = doc["activity"]
    if activity == "deodrant":  # make deodrant and deodorant consistent
        activity = "deodorant"
    elif "RTT" in activity:
        activity = "RTT exercise"
    elif "shelf" in activity:
        activity = "shelf exercise"
    return activity


def sr_id_process_results(doc, results):
    """Process per-document results into metric format"""
    pred_str = results[0]
    pred     = _parse_final_answer(pred_str).lower().strip()
    gt       = sr_id_doc_to_target(doc).lower().strip()
    acc      = (pred == gt)
    return {
        "accuracy": float(acc),
    }


class OutputToResultsFilter:

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
            resp_filtered = ""
            for j in range(len(resps[i][0])):
                payload = resps[i][0][j]
                response, start_time, end_time = payload
                resp_filtered += f"<RESP> {response} <TIME> {start_time:.3f}-{end_time:.3f} "
            resps_filtered.append(resp_filtered)
        return resps_filtered


load_ds = partial(
    load_strokerehab_primitives_dataset,
    patients='all',
    activity='all',
    reps='first',
)
