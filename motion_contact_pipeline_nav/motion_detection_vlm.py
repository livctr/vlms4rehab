#!/usr/bin/env python3
"""
VLM-based windowed motion detection (no RTMPose).

This module windows a video and asks a local VLM (e.g., Qwen2.5-VL) to decide
if the target hand is in motion within each window. It returns per-window
motion, confidence, and rationale, and expands these to frame-level outputs
for downstream integration.

Notes:
- Reuses model loading and vision I/O utilities from contact_detection_vlm.py
- Optionally saves window clips with overlays (wrist-only annotation if coords exist)
"""

import os
import json
import math
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Reuse VLM backend and utilities
import contact_detection_vlm as vlm

# Optional RTMPose wrist extraction (for annotation only)
try:
    from enhanced_rtmpose_analysis import EnhancedRTMPoseExtractor
    _HAS_MMPOSE = True
except Exception:
    _HAS_MMPOSE = False


def build_vlm_system_prompt_motion() -> str:
    return (
        "You are a precise vision-language assistant for rehabilitation movement analysis. "
        "Multiple people may appear in the frames. Focus ONLY on the PATIENT specified in the instructions. "
        "IMPORTANT: Decide if the TARGET HAND is MOVING within the window. Movement includes reaching, transporting, or adjusting; "
        "micro tremors or camera shake do not count. Respond ONLY with a compact JSON object as described."
    )


def build_vlm_user_json_payload_motion(
    activity_ctx: Dict[str, Any],
    handedness: str,
    window_meta: Dict[str, Any],
    wrist_keypoint: Dict[str, Any],
    recent_history: List[Dict[str, Any]],
    wrist_motion_features: Dict[str, float],
    current_step_hint: str = None
) -> Dict[str, Any]:
    return {
        "task": "binary_motion_detection",
        "instructions": {
            "what_to_decide": (
                "Return motion=1 if the TARGET HAND is moving within this window; else motion=0. "
                "Movement means observable displacement of the hand/fingers relative to the body or environment; "
                "ignore minor jitter or compression artifacts."
            ),
            "hand": handedness,
            "definitions": {
                "motion": "Observable purposeful movement of the hand/fingers/palm (including reaching, transporting, repositioning).",
                "no_motion": "Hand is stationary except for negligible tremor/noise.",
            },
            "output_format": {
                "type": "json",
                "schema": {"motion": "0 or 1 integer", "confidence": "0-1 float", "rationale": "<= 200 chars"}
            }
        },
        "activity_context": {
            "name": activity_ctx.get("name"),
            "workspace": activity_ctx.get("workspace"),
            "target_objects": activity_ctx.get("target_objects"),
            "steps": activity_ctx.get("steps", []),
            "current_step_hint": current_step_hint
        },
        "window": window_meta,
        "recent_history": recent_history,
        # Only wrist keypoint per requirement (if available)
        "keypoints": {"wrist": wrist_keypoint} if wrist_keypoint else {"wrist": None},
        "features": wrist_motion_features
    }


def call_vlm_motion_video(video_frames_pil: List[Image.Image], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Local inference using Qwen2.5-VL on a short video sequence for motion detection."""
    if vlm.model is None or (vlm.backend == "qwen" and vlm.processor is None):
        return {"motion": 0, "confidence": 0.0, "rationale": "model_not_loaded"}

    system_prompt = build_vlm_system_prompt_motion()
    user_json_text = json.dumps(payload, ensure_ascii=False)

    try:
        if vlm.backend == "internvl3":
            content = vlm.internvl3_chat_video(video_frames_pil, user_json_text + "\nRespond ONLY with the JSON.")
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "video", "video": video_frames_pil},
                    {"type": "text", "text": user_json_text},
                ]},
            ]
            text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = vlm.process_vision_info(messages)
            inputs = vlm.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")
            generated_ids = vlm.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = vlm.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            content = output_text.strip()
        print(f"VLM motion raw response: {content}")
    except Exception as e:
        return {"motion": 0, "confidence": 0.0, "rationale": f"inference_error: {str(e)}"}

    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start == -1 or end == -1:
            raise json.JSONDecodeError("No JSON object found", content, 0)
        parsed = json.loads(content[start:end])
        motion = int(parsed.get("motion", 0))
        conf = float(parsed.get("confidence", 0.0))
        rationale = str(parsed.get("rationale", ""))
        return {"motion": 1 if motion else 0, "confidence": max(0.0, min(1.0, conf)), "rationale": rationale[:200]}
    except Exception:
        return {"motion": 0, "confidence": 0.0, "rationale": f"parse_error: {content}"}


def draw_wrist_only(frame: np.ndarray, wrist: Dict[str, Any]) -> np.ndarray:
    if not wrist or any(np.isnan([wrist.get('x', np.nan), wrist.get('y', np.nan)])):
        return frame
    wrist_pt = (int(wrist['x']), int(wrist['y']))
    cv2.circle(frame, wrist_pt, 12, (0, 0, 255), -1)
    cv2.circle(frame, wrist_pt, 15, (255, 255, 255), 3)
    cv2.putText(frame, 'WRIST', (wrist_pt[0]-25, wrist_pt[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def write_window_video_with_wrist(
    video_path: str,
    start_t: float,
    end_t: float,
    fps: float,
    width: int,
    height: int,
    output_path: str,
    motion: int,
    confidence: float,
    rationale: str,
    wrist_snapshot: Dict[str, Any]
) -> None:
    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap2.release()
        return
    start_frame = int(start_t * fps)
    end_frame = int(end_t * fps)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    text_lines = [
        f"Motion: {'YES' if motion else 'NO'}",
        f"Confidence: {confidence:.2f}",
        "Rationale:",
    ] + [rationale]
    frame_idx = 0
    for _ in range(start_frame, end_frame):
        ret, frame = cap2.read()
        if not ret:
            break
        frame = draw_wrist_only(frame, wrist_snapshot)
        overlay = frame.copy()
        y_start = height - len(text_lines) * 30 - 20
        cv2.rectangle(overlay, (10, y_start - 10), (520, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        for i, line in enumerate(text_lines):
            y = y_start + i * 28
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if motion else (0, 0, 255), 2)
        out.write(frame)
        frame_idx += 1
    out.release()
    cap2.release()


def _extract_wrist_track(video_path: str, handedness: str, subsample_fps: int = 10) -> Tuple[Dict[int, Dict[str, float]], float]:
    """Extract wrist coordinates at a subsampled FPS; returns (frame_idx->wrist, eff_fps)."""
    if not _HAS_MMPOSE:
        return {}, 0.0
    try:
        extractor = EnhancedRTMPoseExtractor()
        keypoints_data, total_frames_sub, _, _, eff_fps = extractor.extract_keypoints(video_path, subsample_fps)
        # Determine wrist index robustly from names if available
        wrist_index = None
        names = None
        try:
            names = [str(n).lower() for n in (extractor.keypoint_names or [])]
        except Exception:
            names = None
        if names:
            pref = 'left' if handedness.upper() == 'L' else 'right'
            try:
                wrist_index = names.index(f"{pref}_wrist")
            except ValueError:
                wrist_index = None
        if wrist_index is None:
            # Fallback COCO-ish indices
            wrist_index = 9 if handedness.upper() == 'L' else 10
        wrist_track: Dict[int, Dict[str, float]] = {}
        for f_idx in range(total_frames_sub):
            kps = keypoints_data.get(f_idx)
            if kps is None or kps.shape[0] <= wrist_index:
                wrist_track[f_idx] = {"x": float('nan'), "y": float('nan')}
                continue
            x, y, conf = kps[wrist_index]
            wrist_track[f_idx] = {"x": float(x), "y": float(y)}
        return wrist_track, float(eff_fps)
    except Exception:
        return {}, 0.0


def _overlay_wrist_on_pil_frames(frames_pil: List[Image.Image], start_t: float, end_t: float, wrist_track: Dict[int, Dict[str, float]], eff_fps: float) -> List[Image.Image]:
    if not wrist_track or eff_fps <= 0:
        return frames_pil
    n = len(frames_pil)
    if n == 0:
        return frames_pil
    times = np.linspace(start_t, end_t, max(2, n)).tolist()
    out_frames: List[Image.Image] = []
    for i, pil_img in enumerate(frames_pil):
        t = float(times[min(i, len(times)-1)])
        sub_frame_idx = int(round(t * eff_fps))
        wrist = wrist_track.get(sub_frame_idx, {"x": float('nan'), "y": float('nan')})
        # Draw marker using OpenCV
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_bgr = draw_wrist_only(img_bgr, wrist)
        out_frames.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    return out_frames


def _crop_frames_to_wrist_roi(frames_pil: List[Image.Image], start_t: float, end_t: float, wrist_track: Dict[int, Dict[str, float]], eff_fps: float, width: int, height: int, box_px: int = 320) -> List[Image.Image]:
    if not wrist_track or eff_fps <= 0:
        return frames_pil
    n = len(frames_pil)
    if n == 0:
        return frames_pil
    times = np.linspace(start_t, end_t, max(2, n)).tolist()
    cropped: List[Image.Image] = []
    half = box_px // 2
    for i, pil_img in enumerate(frames_pil):
        t = float(times[min(i, len(times)-1)])
        sub_idx = int(round(t * eff_fps))
        wp = wrist_track.get(sub_idx)
        if not wp or any(np.isnan([wp.get('x', np.nan), wp.get('y', np.nan)])):
            cropped.append(pil_img)
            continue
        cx, cy = int(round(wp['x'])), int(round(wp['y']))
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(width, cx + half)
        y2 = min(height, cy + half)
        # Ensure box has positive size
        if x2 - x1 < 10 or y2 - y1 < 10:
            cropped.append(pil_img)
            continue
        cropped.append(pil_img.crop((x1, y1, x2, y2)).resize(pil_img.size))
    return cropped


def _compute_wrist_motion_features(wrist_track: Dict[int, Dict[str, float]], eff_fps: float, start_t: float, end_t: float) -> Dict[str, float]:
    if not wrist_track or eff_fps <= 0 or end_t <= start_t:
        return {"net_disp_px": 0.0, "path_len_px": 0.0, "mean_speed_px_s": 0.0}
    times = np.linspace(start_t, end_t, 8).tolist()
    pts: List[Tuple[float, float]] = []
    for t in times:
        idx = int(round(t * eff_fps))
        wp = wrist_track.get(idx, {"x": float('nan'), "y": float('nan')})
        x, y = float(wp.get('x', float('nan'))), float(wp.get('y', float('nan')))
        pts.append((x, y))
    # Filter NaNs
    pts = [(x, y) for (x, y) in pts if np.isfinite(x) and np.isfinite(y)]
    if len(pts) < 2:
        return {"net_disp_px": 0.0, "path_len_px": 0.0, "mean_speed_px_s": 0.0}
    net = math.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])
    path = 0.0
    for i in range(1, len(pts)):
        path += math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
    duration = max(1e-6, end_t - start_t)
    mean_speed = path / duration
    return {"net_disp_px": float(net), "path_len_px": float(path), "mean_speed_px_s": float(mean_speed)}


def run_vlm_motion_detection(
    video_path: str,
    activities_yaml: str,
    activity: str,
    handedness: str = "L",
    window_s: float = 1.0,
    overlap: float = 0.5,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_frames: int = 16,
    output_csv: str = None,
    window_videos_dir: str = None,
    clear_cache: bool = False,
    low_memory: bool = False,
    subsample_fps: int = 10,
    annotate_wrist: bool = True,
    playback_speed: float = 1.0,
) -> Dict[str, Any]:
    """Run VLM motion detection using windowed video prompts.

    Returns dict with keys: motion_csv, motion_df, fps, total_windows, model_used
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    # Optional cache clear before model load
    if clear_cache:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    # Load VLM model once
    vlm.load_vlm_model(model)

    # Video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Activity context
    activity_ctx = vlm.read_activities_context(activities_yaml, activity)

    # Windows over frames
    windows = vlm.build_windows(total_frames, fps_video, window_s, overlap)

    # Prepare outputs
    results_windows: List[Dict[str, Any]] = []
    past_windows: List[Dict[str, Any]] = []

    if window_videos_dir:
        os.makedirs(window_videos_dir, exist_ok=True)

    # Optional wrist extraction for annotation only
    wrist_track: Dict[int, Dict[str, float]] = {}
    eff_fps = 0.0
    if annotate_wrist and _HAS_MMPOSE:
        wrist_track, eff_fps = _extract_wrist_track(video_path, handedness, subsample_fps=subsample_fps)

    steps_list = activity_ctx.get("steps", []) if isinstance(activity_ctx.get("steps", []), list) else []
    total_windows = max(1, len(windows))
    for w_i, (s_idx, e_idx, s_t, e_t) in enumerate(tqdm(windows, desc="VLM Motion Detection")):
        # Center wrist snapshot: unavailable without pose; set None
        wrist_snapshot = {"x": float('nan'), "y": float('nan')}
        if wrist_track and eff_fps > 0:
            center_t = (s_t + e_t) / 2.0
            sub_idx = int(round(center_t * eff_fps))
            wrist_snapshot = wrist_track.get(sub_idx, wrist_snapshot)

        window_meta = {"start_time_s": float(s_t), "end_time_s": float(e_t), "duration_s": float(max(0.0, e_t - s_t))}
        recent_history = [{
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "motion": r["motion"],
            "rationale": r.get("rationale", "")
        } for r in results_windows[-3:]]

        # Compute numeric wrist motion features for the window
        wrist_feats = _compute_wrist_motion_features(wrist_track, eff_fps, s_t, e_t)
        # Current-step hint by proportional mapping
        cur_step_hint = None
        if steps_list:
            step_idx = int(round((w_i / max(1, total_windows - 1)) * (len(steps_list) - 1)))
            step_idx = min(max(0, step_idx), len(steps_list) - 1)
            cur_step_hint = str(steps_list[step_idx])
        payload = build_vlm_user_json_payload_motion(activity_ctx, handedness, window_meta, wrist_snapshot, recent_history, wrist_feats, current_step_hint=cur_step_hint)

        # Prefer sending multiple frames across the window
        video_frames_pil = vlm.get_window_frames_pil(
            video_path, s_t, e_t, fps_video, width, height,
            max_frames=max(2, int(max_frames)), playback_speed=float(playback_speed)
        )
        if annotate_wrist and wrist_track:
            # Crop to a ROI around the wrist to focus the VLM
            video_frames_pil = _crop_frames_to_wrist_roi(video_frames_pil, s_t, e_t, wrist_track, eff_fps, width, height, box_px=320)
        out = call_vlm_motion_video(video_frames_pil, payload)
        motion = int(out.get("motion", 0))
        confidence = float(out.get("confidence", 0.0))
        rationale = str(out.get("rationale", ""))

        if window_videos_dir:
            video_filename = os.path.join(window_videos_dir, f"motion_{s_t:.1f}s_to_{e_t:.1f}s.mp4")
            write_window_video_with_wrist(video_path, s_t, e_t, fps_video, width, height, video_filename, motion, confidence, rationale, wrist_snapshot)

        results_windows.append({
            "start_time": float(s_t),
            "end_time": float(e_t),
            "motion": motion,
            "confidence": confidence,
            "rationale": rationale
        })

    cap.release()

    # Expand window results to frame-level motion_df
    time_s = np.arange(total_frames, dtype=float) / max(1.0, fps_video)
    frame_motion = np.zeros(total_frames, dtype=int)
    frame_prob = np.zeros(total_frames, dtype=float)
    for (s_idx, e_idx, s_t, e_t), win in zip(windows, results_windows):
        frame_motion[s_idx:e_idx] = int(win["motion"]) if e_idx > s_idx else int(win["motion"])
        frame_prob[s_idx:e_idx] = float(win["confidence"]) if e_idx > s_idx else float(win["confidence"])

    motion_df = pd.DataFrame({
        "frame": np.arange(total_frames, dtype=int),
        "time_s": time_s,
        "prediction": frame_motion.astype(int),
        "probability": np.clip(frame_prob, 0.0, 1.0)
    })

    # Save motion CSV
    if output_csv is None:
        video_identifier = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = f"{video_identifier}_vlm_motion.csv"
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    motion_df.to_csv(output_csv, index=False)

    return {
        "motion_csv": output_csv,
        "motion_df": motion_df,
        "fps": fps_video,
        "total_windows": len(results_windows),
        "model_used": model
    }


