from typing import List, Tuple

import json
from pathlib import Path

from data.utils_strokerehab import string_to_resps

from vidplot.core import StaticDataStreamer

from vidplot.streamers import (
    VideoStreamer,
    TimestampedDataStreamer,
    LabelBarStreamer
)
from vidplot.renderers import (
    RGBRenderer,
    BoxRenderer,
    LabelBarRenderer,
    StringRenderer,
)
from vidplot.core import AnnotationOrchestrator


import json


import os
import glob
import re
from typing import Dict, Any

import os
from typing import Dict, List
from data.utils_strokerehab import DataPaths, PrimitiveLabelUtils
from vidplot.core import StaticDataStreamer, AnnotationOrchestrator
from vidplot.streamers import VideoStreamer, LabelBarStreamer
from vidplot.renderers import StringRenderer, RGBRenderer, LabelBarRenderer



def get_latest_run_files(base_dir: str = "logs") -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Scans a directory of method folders, each containing model subfolders with a single
    inner folder, which holds timestamped run files.
    Returns data[method][model] = {'results': path_to_results, 'samples': path_to_samples}
    for the most recent run.
    """
    data: Dict[str, Dict[str, Dict[str, str]]] = {}
    ts_pattern = re.compile(r"(\d{8}_\d{6})_results\.json$")

    for method in os.listdir(base_dir):
        method_path = os.path.join(base_dir, method)
        if not os.path.isdir(method_path):
            continue
        data[method] = {}

        for model in os.listdir(method_path):
            model_path = os.path.join(method_path, model)
            if not os.path.isdir(model_path):
                continue

            # if there's exactly one subdirectory, search inside it:
            children = [
                os.path.join(model_path, d)
                for d in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, d))
            ]
            search_dir = children[0] if len(children) == 1 else model_path

            runs: Dict[str, Dict[str, str]] = {}
            for res_file in glob.glob(os.path.join(search_dir, "*_results.json")):
                fn = os.path.basename(res_file)
                m = ts_pattern.match(fn)
                if not m:
                    continue
                ts = m.group(1)

                samp_glob = os.path.join(search_dir, f"{ts}_samples_*.jsonl")
                samp_files = glob.glob(samp_glob)
                if not samp_files:
                    continue

                runs[ts] = {
                    "results": res_file,
                    "samples": samp_files[0]
                }

            if runs:
                latest = max(runs)
                data[method][model] = runs[latest]

        # end for model
    # end for method

    return data



def load_sample_results_by_id(jsonl_path: str) -> dict[str, dict]:
    """
    Load a JSONL file where each line is a JSON object containing an 'id' field
    and return a dictionary mapping each record's 'id' to a sub-dictionary of
    selected, easily-accessible fields.
    """
    data_by_id: dict[str, dict] = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rec_id = record.get('id')
            if rec_id is None:
                continue

            data_by_id[rec_id] = {
                'patient': record.get('patient'),
                'activity': record.get('activity'),
                'path_v': record.get('path_v'),
                'fps': record.get('fps'),
                'path_l': record.get('path_l'),
                'target': record.get('target'),
                'resps': record.get('resps'),
                'filtered_resps': record.get('filtered_resps'),
                'edit_score': record.get('edit_score'),
                'action_error_rate': record.get('action_error_rate'),
                'mae': {
                    'reach': record.get('mae_reach'),
                    'reposition': record.get('mae_reposition'),
                    'transport': record.get('mae_transport'),
                    'stabilize': record.get('mae_stabilize'),
                    'idle': record.get('mae_idle'),
                },
                'count': {
                    'reach': record.get('count_reach'),
                    'reposition': record.get('count_reposition'),
                    'transport': record.get('count_transport'),
                    'stabilize': record.get('count_stabilize'),
                    'idle': record.get('count_idle'),
                },
                'count_truth': record.get('count_truth'),
                'count_pred': record.get('count_pred'),
                'input': record.get('input'),
            }
    return data_by_id


def load_results_by_id(log_path):
    log_path = Path(log_path)
    with open(log_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    higher_is_better = data.get("higher_is_better", {})
    configs = data.get("configs", {})

    parsed_metrics = {}

    for task_name, task_results in results.items():
        metric_info = {}
        for k, v in task_results.items():
            if "," in k:
                base_key, _ = k.split(",", 1)
                if base_key.endswith("_stderr"):
                    continue  # we'll pick these up as `*_stderr`
                stderr_key = base_key + "_stderr,combine-and-parse"
                stderr_val = task_results.get(stderr_key, None)

                metric_info[base_key] = {
                    "value": v,
                    "stderr": stderr_val,
                    "higher_is_better": higher_is_better.get(task_name, {}).get(base_key, None)
                }
        parsed_metrics[task_name] = metric_info

    # extract eval-specific flags (optional)
    eval_kwargs = {}
    for task_name, config in configs.items():
        eval_kwargs[task_name] = config.get("lmms_eval_specific_kwargs", {})

    return {
        "metrics": parsed_metrics,
        "eval_kwargs": eval_kwargs,
    }


def parse_yes_segments(
    responses: List[List[List[str]]]
) -> Tuple[List[bool], List[bool], List[float]]:
    """
    Args:
        responses: a list of segment-lists, where each segment is
                   [ "<YES> <SEP> <YES/NO>", "<start_time>", "<end_time>" ]
                   e.g. [
                           [
                             ['Yes <SEP> Yes', '0.0', '0.96'],
                             ['No  <SEP> Yes', '0.96', '1.92'],
                             ...
                           ]
                        ]
    Returns:
        first_flags:  [True, False, ...]   # first “Yes”?
        second_flags: [True, True,  ...]   # second “Yes”?
        timestamps:   [0.0, 0.96, ...]     # start times as floats
    """
    first_flags:   List[bool] = []
    second_flags:  List[bool] = []
    timestamps:    List[float] = []

    for segment_list in responses:
        for text, start_str, _end_str in segment_list:
            # split into the two tokens
            part1, part2 = text.split(" <SEP> ")
            first_flags.append("yes" in part1.strip().lower())
            second_flags.append("yes" in part2.strip().lower())
            timestamps.append(float(start_str))

    return first_flags, second_flags, timestamps




def get_prims_times_for(
    scenario: str,
    scenario_data: Dict[str, Any],
    video_id: str,
    full_label: str
) -> Tuple[List[str], List[float]]:
    """
    Return (primitives, times) for a given scenario key:
      - 'GT': ground truth from full_label
      - 'Ideal' or 'SP': filtered_resps via string_to_resps
      - 'SMC': motion/contact via parse_yes_segments (returns motion labels)
    If missing, returns ([], []).
    """
    if scenario == 'GT':
        prims, times = PrimitiveLabelUtils.convert_labels_to_prims_times(
            full_label, duplicate_last_prim=True
        )
        return [p.lower() for p in prims], times

    entry = scenario_data.get(scenario, {})
    samples = entry.get('samples', {})
    info = samples.get(video_id)
    if not info:
        return [], []

    if scenario in ('Ideal', 'SP'):
        resp_list = info.get('filtered_resps', [])
        if not resp_list:
            return [], []
        prims, times = string_to_resps(resp_list[0], drop_duplicated=False)
        return [p.lower() for p in prims], times

    if scenario == 'SMC':
        resps = info.get('resps', [])
        if not resps:
            return [], []
        motions, contacts, mc_times = PrimitiveLabelUtils.parse_yes_segments(resps[0])
        labels = ['motion' if m else 'no_motion' for m in motions]
        return labels, mc_times

    return [], []


def get_prims_times_motion_contact(label_id: str):
    latest_run_files = get_latest_run_files()
    label_id_to_scenario = {
        "Ideal": "strokerehab_primitives_1",
        "SP": "strokerehab_primitives_2",
        "SMC": "strokerehab_primitives_3"
    }
    motions, contacts, mc_times = [], [], []
    if label_id == "GT":
        sample_info_path = latest_run_files['strokerehab_primitives_1'][MODEL]['samples']
        path_v = load_sample_results_by_id(sample_info_path)[VIDEO_ID]['path_v']
        path_l = load_sample_results_by_id(sample_info_path)[VIDEO_ID]['path_l']
        full_path_v = os.path.join(VID_DIR, path_v)
        full_path_l = os.path.join(DataPaths.RAW_LABEL_DIR, path_l)
        prims, times = PrimitiveLabelUtils.convert_labels_to_prims_times(full_path_l, duplicate_last_prim=True)

    else:  # label_id in ["Ideal", "SP", "SMC"]:
        sample_info_path = latest_run_files[label_id_to_scenario[label_id]][MODEL]['samples']
        path_v = load_sample_results_by_id(sample_info_path)[VIDEO_ID]['path_v']
        path_l = load_sample_results_by_id(sample_info_path)[VIDEO_ID]['path_l']
        full_path_v = os.path.join(VID_DIR, path_v)
        full_path_l = os.path.join(DataPaths.RAW_LABEL_DIR, path_l)
        sample_results = load_sample_results_by_id(sample_info_path)
        info = sample_results[VIDEO_ID]
        resp_list = info.get('resps', [])
        filtered_resp_list = info.get('filtered_resps', [])
        
        prims, times = string_to_resps(filtered_resp_list[0], drop_duplicated=False)
        if label_id == "SMC":
            motions, contacts, mc_times = parse_yes_segments(resp_list[0])
    return full_path_v, full_path_l, prims, times, motions, contacts, mc_times



def build_annotation_layout(
    task: str,
    model: str,
    video_id: str,
    scenarios: Dict[str, str],
    primitive_colors: Dict[str, Tuple[int,int,int]],
    on_off_colors: Dict[str, Tuple[int,int,int]],
    vid_dir: str
) -> AnnotationOrchestrator:
    """
    Assemble an AnnotationOrchestrator with up to 8 rows:
      1: Title
      2: Prompt input
      3: Video
      4: GT text
      5: GT bar
      6: Pred from Ideal text
      7: Ideal bar
      8: Pred from SP text
      9: SP bar
      10: Pred from SMC text
      11: SMC bar
      12: SMC motion text
      13: motion bar
      14: SMC contact text
      15: contact bar
    """
    # Title and prompt use Ideal for metadata
    full_video, full_label, _, _, _, _, _ = get_prims_times_motion_contact(
        'Ideal'
    )
    # start building
    streamers, renderers = [], []
    row = 1
    # Title
    title_txt = f"Video: {video_id} ({PrimitiveLabelUtils.get_handedness(full_label)} hand)"
    ts = StaticDataStreamer('title', title_txt)
    tr = StringRenderer('title', ts, grid_row=(row,row), grid_column=(1,1))
    streamers.append(ts); renderers.append(tr)
    # Prompt
    # row+=1
    # _, _, _, _, _, _, _ = get_prims_times_motion_contact('Ideal')
    # latest_run_files = get_latest_run_files()
    # prompt_txt = load_sample_results_by_id(latest_run_files['strokerehab_primitives_1'][model]['samples'])[video_id]['input']
    # ps = StaticDataStreamer('prompt', prompt_txt)
    # pr = StringRenderer('prompt', ps, grid_row=(row,row), grid_column=(1,1), font_scale=1, font_color=(255,255,255))
    # streamers.append(ps); renderers.append(pr)
    # Video
    row+=1
    vs = VideoStreamer('video', full_video)
    vr = RGBRenderer('video', vs, grid_row=(row,row), grid_column=(1,1))
    video_row = row
    streamers.append(vs); renderers.append(vr)
    # Helper to add text+bar
    def add_bars(label_id, texts, color_map):
        nonlocal row
        fv, fl, prims, times, motions, contacts, mc_times = get_prims_times_motion_contact(
            label_id
        )
        labels = [prims, motions, contacts]
        times = [times, mc_times, mc_times]

        for text, label, time in zip(texts, labels, times):
            if len(label) > 0 and isinstance(label[0], str):
                # Convert to lower case if it's a string label
                label = [l.lower() for l in label]
            elif len(label) > 0 and isinstance(label[0], bool):
                # Convert boolean labels to 'motion'/'no_motion' or 'contact'/'no_contact'
                label = ['motion' if l else 'no_motion' for l in label] if 'Motion' in text else \
                        ['contact' if l else 'no_contact' for l in label]

            # text
            row += 1
            tsx = StaticDataStreamer(f"txt_{text}", text)
            trx = StringRenderer(f"txt_{text}", tsx, grid_row=(row,row), grid_column=(1,1))
            streamers.append(tsx); renderers.append(trx)
            # bar
            row += 1
            lbs = LabelBarStreamer(f"bar_{text}", {'Time_s': time, label_id: label}, label_id, 'Time_s', duration=vs.duration)
            lbr = LabelBarRenderer(f"bar_{text}", lbs, label_to_color=color_map, grid_row=(row,row), grid_column=(1,1), progress_bar_color=(0,0,0))
            streamers.append(lbs); renderers.append(lbr)

    # GT
    add_bars('GT', ['GT'], primitive_colors)
    # Ideal
    add_bars('Ideal', ['Pred from Ideal Prompt'], primitive_colors)
    # SP
    add_bars('SP', ['Pred from Single Prediction Prompt'], primitive_colors)
    # SMC primitives
    add_bars('SMC', ['Pred from Motion & Contact', 'Motion', 'Contact'], primitive_colors)
    # orchestrator
    grid_rows = [20] * (video_row-1) + [vs.size[1]] + [20]*(row-video_row)
    grid_cols = [max(vs.size[0],600)]
    orch = AnnotationOrchestrator(grid_template_rows=grid_rows, grid_template_columns=grid_cols, gap=0)
    orch.set_annotators(streamers=streamers, renderers=renderers, routes=[(s.name, r.name) for s,r in zip(streamers, renderers)])
    return orch


if __name__ == '__main__':
    PRIMITIVE_COLORS = {
        "reach": (255, 165, 0),      # Orange — active, attention-grabbing, reaching out
        "reposition": (0, 191, 255), # Deep Sky Blue — movement and adjustment, fluidity
        "transport": (34, 139, 34),  # Forest Green — stability in motion, deliberate progress
        "stabilize": (75, 0, 130),   # Indigo — control, quiet strength, holding steady
        "idle": (169, 169, 169),     # Dark Gray — inactive, neutral, waiting
        "motion": (34, 139, 34),
        "no_motion": (169, 169, 169),
        "contact": (34, 139, 34),
        "no_contact": (169, 169, 169),
    }
    ON_OFF_COLORS = {
    }

    USE_SEG = False
    LABEL_IDS = ["GT", "Ideal", "SP", "SMC"]
    VIDEO_ID = 'S00042_combing3_2'
    MODEL = 'nvila_8b'
    VID_DIR = DataPaths.RAW_VIDEO_DIR

    orch = build_annotation_layout(
        task='strokerehab_primitives_1', model=MODEL, video_id=VIDEO_ID,
        scenarios={}, primitive_colors=PRIMITIVE_COLORS, on_off_colors=ON_OFF_COLORS,
        vid_dir=VID_DIR
    )

    orch.show_layout('postprocess/plots/primitives/test_layout.png')
    orch.write('postprocess/plots/primitives/test.png')
