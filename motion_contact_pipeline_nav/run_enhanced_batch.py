#!/usr/bin/env python3
import os
import random
import json
import csv
import subprocess
import sys

import pandas as pd

try:
    from utils import LabelUtils as _LabelUtils
except Exception:
    _LabelUtils = None

BASE_DATA_PATH = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/"
METADATA_CSV_PATH = os.path.join(os.path.dirname(__file__), "cleaned_metadata.csv")
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "my_video_analysis_output_working", "enhanced_batch_results")

# Helper to build paths consistent with main.py comments
VIDEO_PREFIX = os.path.join(BASE_DATA_PATH, "VideoData", "rawVideosADLsandFM")
LABEL_PREFIX = os.path.join(BASE_DATA_PATH, "rawVideoLabels")

ENHANCED_SCRIPT = os.path.join(os.path.dirname(__file__), "enhanced_rtmpose_analysis.py")
VIDEO_GEN_SCRIPT = os.path.join(os.path.dirname(__file__), "enhanced_video_generator.py")
CONTACT_SCRIPT = os.path.join(os.path.dirname(__file__), "contact_detection_vlm.py")
ACTIVITIES_YAML = os.path.join(os.path.dirname(__file__), "activities_ground_truth.yaml")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sample_videos(n: int = 10):
    if not os.path.exists(METADATA_CSV_PATH):
        print(f"Metadata CSV not found at {METADATA_CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(METADATA_CSV_PATH)
    df = df.sample(n=min(n, len(df)))
    entries = []
    for _, row in df.iterrows():
        video_id = row['id']
        video_path = os.path.join(VIDEO_PREFIX, row['path_v'])
        label_path = os.path.join(LABEL_PREFIX, row['path_l']) if pd.notna(row['path_l']) else None
        activity = str(row.get('activity', 'unknown'))
        entries.append((video_id, video_path, label_path, activity))
    return entries


def _detect_hand(label_path: str) -> str:
    if label_path and os.path.exists(label_path) and _LabelUtils is not None:
        try:
            hand = _LabelUtils.get_handedness(label_path)
            if str(hand).lower().startswith('l'):
                return 'L'
            if str(hand).lower().startswith('r'):
                return 'R'
        except Exception:
            pass
    return 'L'


def run_enhanced_for_video(video_id: str, video_path: str, label_path: str, out_dir: str, subsample_fps: int = 10, algo: str = "hybrid", window_s: float = 1.0, overlap: float = 0.5, thresh_method: str = 'percentile', percentile: float = 0.75, mad_k: float = 1.5, activity: str = "unknown"):
    ensure_dir(out_dir)
    handedness = _detect_hand(label_path)
    cmd = [
        sys.executable, ENHANCED_SCRIPT,
        "--video_path", video_path,
        "--label_path", label_path,
        "--handedness", handedness,
        "--output_dir", out_dir,
        "--subsample_fps", str(subsample_fps),
        "--algo", algo
    ]
    if algo == 'windowed':
        cmd += [
            "--window_s", str(window_s),
            "--overlap", str(overlap),
            "--thresh_method", str(thresh_method),
            "--percentile", str(percentile),
            "--mad_k", str(mad_k)
        ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Find the motion CSV just written
    csv_candidates = [f for f in os.listdir(out_dir) if f.endswith("_enhanced_motion_data.csv")]
    if not csv_candidates:
        print(f"No motion CSV found in {out_dir}")
        return None
    motion_csv = os.path.join(out_dir, csv_candidates[0])

    # Determine analysis FPS from motion CSV time_s spacing
    try:
        mdf = pd.read_csv(motion_csv, usecols=['time_s'])
        if len(mdf) >= 2:
            dt = float(pd.Series(mdf['time_s'].values).diff().median())
            analysis_fps = int(round(1.0 / dt)) if dt and dt > 0 else subsample_fps
        else:
            analysis_fps = subsample_fps
    except Exception:
        analysis_fps = subsample_fps

    # Optional: if original video is accessible, create overlay video
    out_video = os.path.join(out_dir, f"{video_id}_enhanced_overlay.mp4")
    cmd2 = [
        sys.executable, VIDEO_GEN_SCRIPT,
        "--motion_data", motion_csv,
        "--video_path", video_path,
        "--handedness", handedness,
        "--output_path", out_video,
        "--subsample_fps", str(analysis_fps)
    ]
    print("Generating overlay:", " ".join(cmd2))
    try:
        subprocess.run(cmd2, check=True)
    except subprocess.CalledProcessError as e:
        print("Overlay generation failed:", e)

    # Run contact detection per window using VLM (Qwen2.5-VL)
    contact_csv = os.path.join(out_dir, f"{video_id}_window_contact.csv")
    cmd3 = [
        sys.executable, CONTACT_SCRIPT,
        "--motion_csv", motion_csv,
        "--video_path", video_path,
        "--activities_yaml", ACTIVITIES_YAML,
        "--activity", activity,
        "--handedness", handedness,
        "--window_s", str(window_s),
        "--overlap", str(overlap),
        "--model", "OpenGVLab/InternVL3-38B",
        "--internvl_split",
        "--max_frames", "32",
        "--window_videos_dir", os.path.join(out_dir, "window_videos"),
        "--output_csv", contact_csv
    ]
    print("Contact detection:", " ".join(cmd3))
    try:
        subprocess.run(cmd3, check=True)
    except subprocess.CalledProcessError as e:
        print("Contact detection failed:", e)

    # Load metrics JSON
    metrics_json_file = [f for f in os.listdir(out_dir) if f.endswith("_metrics.json")]
    metrics = None
    if metrics_json_file:
        with open(os.path.join(out_dir, metrics_json_file[0]), 'r') as jf:
            metrics = json.load(jf)
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run enhanced RTMPose analysis in batch")
    parser.add_argument("--n", type=int, default=5, help="Number of videos to sample")
    parser.add_argument("--fps", type=int, default=30, help="Subsample FPS for analysis")
    parser.add_argument("--algo", type=str, default="windowed", choices=["hmm", "hybrid", "windowed"], help="Motion detection algorithm")
    parser.add_argument("--window_s", type=float, default=1.0, help="Window size in seconds for dynamic thresholding")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction (0-1)")
    parser.add_argument("--thresh_method", type=str, default='percentile', choices=['percentile','mad'], help="Dynamic threshold method")
    parser.add_argument("--percentile", type=float, default=0.75, help="Percentile for percentile-based thresholding")
    parser.add_argument("--mad_k", type=float, default=1.5, help="K multiplier for MAD-based thresholding")
    args = parser.parse_args()

    ensure_dir(RESULTS_ROOT)
    entries = sample_videos(args.n)

    summary = []
    for i, (video_id, video_path, label_path, activity) in enumerate(entries, 1):
        vid_dir = os.path.join(RESULTS_ROOT, video_id)
        try:
            metrics = run_enhanced_for_video(
                video_id, video_path, label_path, vid_dir,
                subsample_fps=args.fps, algo=args.algo,
                window_s=args.window_s, overlap=args.overlap,
                thresh_method=args.thresh_method, percentile=args.percentile, mad_k=args.mad_k,
                activity=activity
            )
            if metrics:
                metrics['activity'] = activity
                summary.append(metrics)
        except subprocess.CalledProcessError as e:
            print(f"Failed on {video_id}: {e}")
            continue

    # Save aggregated metrics
    summary_json = os.path.join(RESULTS_ROOT, "summary_metrics.json")
    with open(summary_json, 'w') as jf:
        json.dump(summary, jf, indent=2)

    # Save CSV with averages at bottom
    if summary:
        keys = sorted({k for m in summary for k in m.keys()})
        csv_path = os.path.join(RESULTS_ROOT, "summary_metrics.csv")
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=keys)
            writer.writeheader()
            for m in summary:
                writer.writerow(m)

        # Compute averages for numeric fields
        import math
        agg = {}
        for k in keys:
            vals = [m[k] for m in summary if isinstance(m.get(k), (int, float)) and not math.isnan(m.get(k))]
            agg[k] = sum(vals) / len(vals) if vals else None
        with open(os.path.join(RESULTS_ROOT, "summary_averages.json"), 'w') as jf:
            json.dump(agg, jf, indent=2)
        print("Averages:", agg)
    print(f"Done. Results in {RESULTS_ROOT}")

if __name__ == "__main__":
    main() 