#!/usr/bin/env python3
"""
Basic IDLE and CONTACT experiments for stroke rehabilitation analysis.

This script runs the basic IDLE and CONTACT experiments as described in the research plan:
- Patients: C00020 (healthy), S0001 (mild), S0005 (moderate), S00021 (severe)
- Activities: every activity, first repetition, both views
- Models: Qwen2.5-VL
- Metrics: accuracy and F1 score

Contact: transport/stabilize are positive, idle/reposition/reach are negative.
Idle: ONLY count idle/reposition/reach. DO NOT include transport/stabilize.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import cv2

# Import our modules
from lmms_eval.tasks.strokerehab.utils_primitives import load_strokerehab_primitives_dataset
from data.utils_strokerehab import DataPaths, PrimitiveLabelUtils
from tools.ultralytics_pose import Pose2DStream
from tools.signal_generator import predict_with_state_machine, IDLE_PROMPT_METHODS, CONTACT_PROMPT_METHODS
from tools.vqa.qwen2_5_vl import Qwen2_5_VL_VQA


class DummyVLM:
    """Dummy VLM for testing without GPU."""
    def process_frames(self, frames, prompt) -> str:
        print(f"DUMMY VLM PROMPT: {prompt}")
        if "Locate" in prompt:
            return """
[
  {"bbox_2d": [100, 150, 300, 400], "label": "person"},
  {"bbox_2d": [120, 180, 220, 280], "label": "hand"},
  {"bbox_2d": [50, 60, 100, 120], "label": "cup"}
]
"""
        elif "idle" in prompt.lower():
            return "No."  # Assume not idle for testing
        elif "contact" in prompt.lower():
            return "Yes."  # Assume contact for testing
        return "IDK"

    def clear(self):
        pass


def get_experiment_config() -> Dict[str, Any]:
    """Get the experiment configuration."""
    return {
        "patients": ["C00020", "S0001", "S0005", "S00021"],
        "video_regex": r'^.*/(C00020|S0001|S0005|S00021)_.*1_[1]\.(mkv|avi)$',
        "idle_prompt_methods": ["SMC", "Idle", "StatefulIdleFromPred", "StatefulIdleFromGT", "Focus"],
        "contact_prompt_methods": ["SMC", "Basic", "StatefulContactFromPred", "StatefulContactFromGT"],
        "crop_methods": ["window", "tracklet"],
        "chunk_max_frames": 4,
        "sampling_fps": 15,
        "use_dummy_vlm": False,  # Set to False to use real Qwen model
    }


def load_dataset(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Load the dataset for the specified patients."""
    print("Loading dataset...")
    ds = load_strokerehab_primitives_dataset(
        video_regex=config["video_regex"]
    )
    paths = pd.DataFrame(ds['test'])[['path_v', 'path_l']]
    
    path_ls = [os.path.join(DataPaths.RAW_LABEL_DIR, p) for p in paths['path_l'].tolist()]
    path_vs = [os.path.join(DataPaths.RAW_VIDEO_DIR, p) for p in paths['path_v'].tolist()]
    
    print(f"Number of videos: {len(path_vs)}")
    print(f"First three: {path_vs[:3]}")
    
    return path_vs, path_ls


def setup_vlm(config: Dict[str, Any]):
    """Setup the VLM (either dummy or real Qwen model)."""
    if config["use_dummy_vlm"]:
        print("Using dummy VLM for testing...")
        return DummyVLM()
    else:
        print("Loading Qwen2.5-VL model...")
        return Qwen2_5_VL_VQA(
            pretrained="Qwen/Qwen2.5-VL-32B-Instruct",
            device="cuda",
            device_map=None,
            use_cache=True,
        )


def get_ground_truth_labels(label_path: str, start_t: float, end_t: float) -> Tuple[bool, bool]:
    """
    Get ground truth labels for a time window.
    
    Returns:
        gt_idle: True if the window contains only idle/reposition/reach
        gt_contact: True if the window contains transport/stabilize
    """
    gt_prims, gt_times = PrimitiveLabelUtils.convert_labels_to_prims_times(label_path)
    
    # Find the primitive that covers the majority of the time window
    window_center = (start_t + end_t) / 2
    
    # Find the primitive at the window center
    for i in range(len(gt_prims)):
        if gt_times[i] <= window_center < gt_times[i + 1]:
            primitive = gt_prims[i]
            break
    else:
        # If window is beyond the last primitive, use the last one
        primitive = gt_prims[-1]
    
    # Contact: transport/stabilize are positive, others are negative
    gt_contact = primitive in ("transport", "stabilize")
    
    # Idle: only count idle/reposition/reach (exclude transport/stabilize)
    # gt_idle = primitive in ("idle", "reposition", "reach")
    gt_idle = primitive in ("idle")
    
    return gt_idle, gt_contact


def run_single_video_experiment(
    video_path: str,
    label_path: str,
    handedness: str,
    vlm,
    pose_stream: Pose2DStream,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Run experiment on a single video."""
    print(f"Processing {video_path}...")
    
    # Run the state machine
    df = predict_with_state_machine(
        video_path=video_path,
        label_path=label_path,
        handedness=handedness,
        vlm=vlm,
        pose_stream=pose_stream,
        chunk_max_frames=config["chunk_max_frames"],
        sampling_fps=config["sampling_fps"],
        include_prompt_info_in_df=True,
        idle_prompt_methods=config["idle_prompt_methods"],
        contact_prompt_methods=config["contact_prompt_methods"],
        crop_methods=config["crop_methods"]
    )
    
    # Add ground truth labels
    gt_idles = []
    gt_contacts = []
    
    for _, row in df.iterrows():
        gt_idle, gt_contact = get_ground_truth_labels(
            label_path, row['start_t'], row['end_t']
        )
        gt_idles.append(gt_idle)
        gt_contacts.append(gt_contact)
    
    df['gt_idle'] = gt_idles
    df['gt_contact'] = gt_contacts
    
    return df


def calculate_metrics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Calculate accuracy and F1 scores for all methods."""
    metrics = {}
    
    # Calculate metrics for each crop method
    for crop_method in config["crop_methods"]:
        metrics[crop_method] = {}
        
        # Idle metrics
        for method in config["idle_prompt_methods"]:
            col_name = f"{crop_method}_{method}"
            if col_name in df.columns:
                y_true = df['gt_idle'].astype(int)
                y_pred = df[col_name].astype(int)
                
                # Only evaluate on samples where gt_idle is relevant (idle/reposition/reach)
                # This means we exclude transport/stabilize from idle evaluation
                relevant_mask = df['gt_idle'] | (df['gt_contact'] == False)
                if relevant_mask.sum() > 0:
                    y_true_rel = y_true[relevant_mask]
                    y_pred_rel = y_pred[relevant_mask]
                    
                    accuracy = accuracy_score(y_true_rel, y_pred_rel)
                    f1 = f1_score(y_true_rel, y_pred_rel, average='binary')
                    
                    metrics[crop_method][f"idle_{method}"] = {
                        "accuracy": accuracy,
                        "f1": f1,
                        "n_samples": relevant_mask.sum()
                    }
        
        # Contact metrics
        for method in config["contact_prompt_methods"]:
            col_name = f"{crop_method}_{method}"
            if col_name in df.columns:
                y_true = df['gt_contact'].astype(int)
                y_pred = df[col_name].astype(int)
                
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='binary')
                
                metrics[crop_method][f"contact_{method}"] = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "n_samples": len(y_true)
                }
    
    return metrics


def print_results(metrics: Dict[str, Dict[str, float]]):
    """Print the experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    
    for crop_method, crop_metrics in metrics.items():
        print(f"\n{crop_method.upper()} CROP METHOD:")
        print("-" * 40)
        
        # Idle results
        print("\nIDLE DETECTION:")
        for key, values in crop_metrics.items():
            if key.startswith("idle_"):
                method = key.replace("idle_", "")
                print(f"  {method:20s}: Acc={values['accuracy']:.3f}, F1={values['f1']:.3f}, N={values['n_samples']}")
        
        # Contact results
        print("\nCONTACT DETECTION:")
        for key, values in crop_metrics.items():
            if key.startswith("contact_"):
                method = key.replace("contact_", "")
                print(f"  {method:20s}: Acc={values['accuracy']:.3f}, F1={values['f1']:.3f}, N={values['n_samples']}")


def _collect_overlay_keys(
    df_video: pd.DataFrame,
    config: Dict[str, Any],
    crop_method: str,
    user_keys: Optional[List[str]] = None,
) -> List[str]:
    """Decide which columns to overlay as rows (default to 9 keys)."""
    if user_keys:
        keys = []
        for k in user_keys:
            # Allow short names like "SMC" that will be prefixed
            if k in ("gt_idle", "gt_contact"):
                if k in df_video.columns:
                    keys.append(k)
            else:
                col = f"{crop_method}_{k}"
                if col in df_video.columns:
                    keys.append(col)
        return keys

    keys: List[str] = []
    for meta in (f"{crop_method}_should_infer", f"{crop_method}_moving_tracklet", f"{crop_method}_other_hand_in_view"):
        if meta in df_video.columns:
            keys.append(meta)
    for m in ("Idle", "SMC"):
        col = f"{crop_method}_{m}"
        if col in df_video.columns:
            keys.append(col)
    for m in ("Basic", "SMC"):
        col = f"{crop_method}_{m}"
        if col in df_video.columns and col not in keys:
            keys.append(col)
    if 'gt_idle' in df_video.columns:
        keys.append('gt_idle')
    if 'gt_contact' in df_video.columns:
        keys.append('gt_contact')
    return keys[:9]


def _val_to_str(x: Any) -> str:
    if isinstance(x, (bool, np.bool_)):
        return 'True' if bool(x) else 'False'
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'nan'
    return str(x)


def _value_at_time(df_video: pd.DataFrame, key: str, t: float) -> Any:
    rows = df_video[(df_video['start_t'] <= t) & (df_video['end_t'] > t)]
    if len(rows) == 0:
        rows = df_video[df_video['start_t'] <= t].tail(1)
    if len(rows) == 0:
        return None
    return rows.iloc[0][key]


def _render_timeline_background_cv(
    df_video: pd.DataFrame,
    crop_method: str,
    keys: List[str],
    width_px: int,
    row_height: int = 26,
    label_gutter_px: int = 210,
    margin_px: int = 6,
) -> Tuple[np.ndarray, float, float, List[int], Dict[str, Tuple[int, int, int]]]:
    """Create a static RGB image with colored spans per key using only OpenCV."""
    dfv = df_video.sort_values('start_t').reset_index(drop=True)
    t_min = float(dfv['start_t'].min())
    t_max = float(dfv['end_t'].max())

    n_rows = len(keys)
    height_px = margin_px * 2 + n_rows * row_height
    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    # Build palette on-the-fly
    palette_colors = [
        (230, 57, 70), (29, 53, 87), (69, 123, 157), (168, 218, 220), (42, 157, 143),
        (233, 196, 106), (244, 162, 97), (231, 111, 81), (87, 117, 144), (67, 170, 139),
        (249, 199, 79), (243, 114, 44), (137, 63, 69), (61, 64, 91), (76, 201, 240),
    ]
    value_to_color: Dict[str, Tuple[int, int, int]] = {}

    def _get_color_for_value(vs: str) -> Tuple[int, int, int]:
        if vs not in value_to_color:
            value_to_color[vs] = palette_colors[len(value_to_color) % len(palette_colors)][::-1]
        return value_to_color[vs]

    # Draw each row
    row_centers: List[int] = []
    for i, key in enumerate(keys):
        y0 = margin_px + i * row_height
        y1 = y0 + row_height - 1
        row_centers.append((y0 + y1) // 2)
        # Row label background
        cv2.rectangle(img, (0, y0), (label_gutter_px - 1, y1), (245, 245, 245), thickness=-1)
        cv2.rectangle(img, (0, y0), (label_gutter_px - 1, y1), (220, 220, 220), thickness=1)
        label = key
        try:
            cv2.putText(img, label, (8, y0 + row_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception:
            pass

        # Draw spans
        starts = dfv['start_t'].values
        ends = dfv['end_t'].values
        vals = dfv[key].values
        for s, e, v in zip(starts, ends, vals):
            xs = int(round(label_gutter_px + (0 if t_max <= t_min else (float(s) - t_min) / (t_max - t_min) * (width_px - label_gutter_px - 1))))
            xe = int(round(label_gutter_px + (0 if t_max <= t_min else (float(e) - t_min) / (t_max - t_min) * (width_px - label_gutter_px - 1))))
            xs = max(label_gutter_px, min(width_px - 1, xs))
            xe = max(xs + 1, min(width_px, xe))
            col = _get_color_for_value(_val_to_str(v))
            cv2.rectangle(img, (xs, y0 + 3), (xe, y1 - 3), col, thickness=-1)

        # Row separator
        cv2.line(img, (label_gutter_px, y1), (width_px - 1, y1), (225, 225, 225), 1)

    # Override colors for booleans (purple -> green, brown -> red requested)
    value_to_color['True'] = (0, 200, 0)   # green (BGR)
    value_to_color['False'] = (0, 0, 255)  # red (BGR)

    # Outer border
    cv2.rectangle(img, (0, 0), (width_px - 1, height_px - 1), (210, 210, 210), 1)

    return img, t_min, t_max, row_centers, value_to_color


def _draw_cursor_and_values(
    base_img: np.ndarray,
    t_cur: float,
    t_min: float,
    t_max: float,
    keys: List[str],
    row_centers: List[int],
    values_at_time: List[str],
    label_gutter_px: int = 210,
) -> np.ndarray:
    out = base_img.copy()
    h, w, _ = out.shape
    # Cursor position
    if t_max <= t_min:
        x = w - 2
    else:
        frac = (t_cur - t_min) / (t_max - t_min)
        x = int(round(label_gutter_px + frac * (w - label_gutter_px - 1)))
        x = max(label_gutter_px, min(w - 2, x))
    cv2.line(out, (x, 1), (x, h - 2), (0, 0, 255), 2)
    return out


def _draw_info_panel_top_left(
    frame_bgr: np.ndarray,
    keys: List[str],
    values_at_time: List[str],
    *,
    x0: int = 6,
    y0: int = 6,
) -> np.ndarray:
    """Draw the info box in the top-left corner of the video frame (BGR)."""
    out = frame_bgr.copy()
    h, w, _ = out.shape
    panel_w = min(260, max(180, w // 6))
    line_h = 18
    panel_h = 12 + line_h * len(values_at_time) + 12
    y1 = min(h - 6, y0 + panel_h)
    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y1), (255, 255, 255), thickness=-1)
    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y1), (0, 0, 0), thickness=1)
    for i, (key, val) in enumerate(zip(keys, values_at_time)):
        y_text = y0 + 22 + i * line_h
        cv2.putText(out, f"{key.split('_')[-1]}: {val}", (x0 + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def generate_timeline_overlay_video(
    df_video: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str,
    crop_method: str,
    keys: Optional[List[str]] = None,
    fps_override: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> None:
    if df_video.empty:
        return
    video_path = df_video['video_path'].iloc[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: cannot open video: {video_path}")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    fps = float(fps_override) if fps_override else float(src_fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Decide keys and render base overlay image
    sel_keys = _collect_overlay_keys(df_video, config, crop_method, user_keys=keys)
    base_img, t_min, t_max, row_centers, _palette = _render_timeline_background_cv(
        df_video, crop_method, sel_keys, width_px=w, row_height=26, label_gutter_px=210, margin_px=6
    )
    th, tw = base_img.shape[:2]

    # Video writer
    out_dir = os.path.join(output_dir, "video_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{Path(video_path).stem}__{crop_method}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h + th))
    print(f"Generating timeline overlay video: {out_path}")

    frame_idx = 0
    success, frame_bgr = cap.read()
    while success:
        t_cur = frame_idx / fps
        cur_vals: List[str] = []
        for key in sel_keys:
            v = _value_at_time(df_video, key, t_cur)
            cur_vals.append(_val_to_str(v))
        overlay = _draw_cursor_and_values(base_img, t_cur, t_min, t_max, sel_keys, row_centers, cur_vals)

        if frame_bgr.shape[1] != w or frame_bgr.shape[0] != h:
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        # Draw info panel on the video frame (top-left of the full output)
        frame_bgr = _draw_info_panel_top_left(frame_bgr, sel_keys, cur_vals)
        stacked = np.vstack([frame_bgr, overlay])  # both are BGR
        writer.write(stacked)

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break
        success, frame_bgr = cap.read()

    writer.release()
    cap.release()
    print(f"Saved overlay video to {out_path}")

def save_results(results: Dict[str, Any], output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Save metrics
    metrics_serializable = convert_numpy_types(results["metrics"])
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save detailed dataframe
    results["combined_df"].to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run basic IDLE and CONTACT experiments")
    parser.add_argument("--output-dir", default="./experiment_results", help="Output directory for results")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--use-real-vlm", action="store_true", help="Use real Qwen model instead of dummy")
    parser.add_argument("--patient", type=str, help="Process only specific patient (C00020, S0001, S0005, S00021)")
    parser.add_argument("--video-overlay", action="store_true", help="Generate timeline overlay videos with cursor")
    parser.add_argument("--overlay-crops", nargs='*', default=None, help="Crop methods to render (default: all in config)")
    parser.add_argument("--overlay-keys", nargs='*', default=None, help="Optional list of short keys to overlay (e.g., Idle SMC Basic gt_idle gt_contact)")
    parser.add_argument("--overlay-fps", type=float, default=None, help="Override output FPS for overlay videos")
    parser.add_argument("--overlay-max-frames", type=int, default=None, help="Max frames per overlay video (for smoke tests)")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_experiment_config()
    if args.use_real_vlm:
        config["use_dummy_vlm"] = False
    if args.patient:
        config["patients"] = [args.patient]
        config["video_regex"] = f'^.*/({args.patient})_.*1_[12]\\.(mkv|avi)$'
    
    print("Basic IDLE and CONTACT Experiments")
    print("="*50)
    print(f"Patients: {config['patients']}")
    print(f"Using {'real' if not config['use_dummy_vlm'] else 'dummy'} VLM")
    print(f"Max videos: {args.max_videos or 'all'}")
    
    # Load dataset
    path_vs, path_ls = load_dataset(config)
    
    if args.max_videos:
        path_vs = path_vs[:args.max_videos]
        path_ls = path_ls[:args.max_videos]
    
    # Setup VLM and pose stream
    vlm = setup_vlm(config)
    pose_stream = Pose2DStream()
    
    # Run experiments
    all_results = []
    combined_df = None
    
    for i, (video_path, label_path) in enumerate(zip(path_vs, path_ls)):
        print(f"\nProcessing video {i+1}/{len(path_vs)}")
        
        # Get handedness from label
        handedness = PrimitiveLabelUtils.get_handedness(label_path)
        
        # Run experiment
        df = run_single_video_experiment(
            video_path, label_path, handedness, vlm, pose_stream, config
        )
        
        # Add video info
        df['video_path'] = video_path
        df['label_path'] = label_path
        df['handedness'] = handedness
        
        # Combine results
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        # Clear VLM state
        vlm.clear()
        pose_stream.clear(keep_slot_labels=False, keep_pending_prompts=False)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(combined_df, config)
    
    # Print results
    print_results(metrics)
    
    # Save results
    results = {
        "metrics": metrics,
        "combined_df": combined_df,
        "config": config
    }
    save_results(results, args.output_dir)
    
    # Optional: generate timeline overlay videos
    if args.video_overlay:
        print("\n" + "="*80)
        print("GENERATING TIMELINE OVERLAY VIDEOS")
        print("="*80)
        crops = args.overlay_crops if args.overlay_crops else config["crop_methods"]
        for video_path in combined_df['video_path'].unique():
            df_vid = combined_df[combined_df['video_path'] == video_path]
            for crop in crops:
                generate_timeline_overlay_video(
                    df_vid,
                    config,
                    args.output_dir,
                    crop_method=crop,
                    keys=args.overlay_keys,
                    fps_override=args.overlay_fps,
                    max_frames=args.overlay_max_frames,
                )

    print(f"\nExperiment completed! Processed {len(path_vs)} videos.")


if __name__ == "__main__":
    main()
