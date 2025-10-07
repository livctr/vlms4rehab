#!/usr/bin/env python3
"""
Windowed contact detection using a VLM (Qwen2.5-VL-7B-Instruct).

For each video and its motion analysis CSV, we:
- Build sliding windows (window_s, overlap)
- Aggregate motion per window (majority of frames)
- For each window, construct a JSON prompt with: activity context, hand, target objects, and keypoints
- Send one representative frame per window (center) to the VLM with the JSON prompt
- Parse VLM JSON to get contact (0/1)
- Save CSV: start_time, end_time, motion, contact

Backend: Direct local inference using Hugging Face transformers with official Qwen2.5-VL approach.
"""

import os
import argparse
import json
import base64
import io
import math
import yaml
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d, median_filter

# Local VLM inference dependencies
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import textwrap
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Global model, tokenizer, and processor loaded once
model = None
processor = None
backend = None  # "qwen" or "internvl3"
tokenizer = None
current_model_id = None


def build_transform(input_size: int):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[Tuple[int, int]], width: int, height: int, image_size: int):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def split_model_device_map(model_id: str) -> Dict[str, int]:
    device_map: Dict[str, int] = {}
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return device_map
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # InternVL3 exposes llm_config under config; fallback safely
    num_layers = getattr(getattr(config, 'llm_config', config), 'num_hidden_layers', None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        return device_map
    # First GPU hosts ViT and part of LLM; treat as half GPU
    per = math.ceil(num_layers / max(1, (world_size - 0.5)))
    per_list = [per] * world_size
    per_list[0] = math.ceil(per_list[0] * 0.5)
    layer_cnt = 0
    for i, n in enumerate(per_list):
        for _ in range(n):
            if layer_cnt >= num_layers:
                break
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    for k in [
        'language_model.model.tok_embeddings',
        'language_model.model.embed_tokens',
        'language_model.output',
        'language_model.model.norm',
        'language_model.model.rotary_emb',
        'language_model.lm_head',
        f'language_model.model.layers.{max(0, num_layers - 1)}',
    ]:
        device_map[k] = 0
    return device_map


def load_vlm_model(model_id: str, internvl_quant: str = 'none', internvl_split: bool = False):
    """Load VLM backend. Supports Qwen2.5-VL and InternVL3-78B."""
    global model, processor, tokenizer, backend, current_model_id
    # If same model already loaded, skip
    if model is not None and current_model_id == model_id:
        return
    # If a different model is loaded, unload and clear cache
    if model is not None and current_model_id != model_id:
        try:
            del model
            if processor is not None:
                del processor
            if tokenizer is not None:
                del tokenizer
            model = None
            processor = None
            tokenizer = None
            backend = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"Unloaded previous VLM model: {current_model_id}")
        except Exception:
            pass

    print(f"Loading local VLM model: {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA device found. VLM inference will be extremely slow.")

    try:
        if "OpenGVLab/InternVL3" in model_id or "InternVL3" in model_id:
            backend = "internvl3"
            kwargs = dict(
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            if internvl_split and torch.cuda.device_count() > 1:
                kwargs["device_map"] = split_model_device_map(model_id)
            # Always load full precision (no 8-bit quantization)
            model = AutoModel.from_pretrained(model_id, **kwargs).eval()
            if torch.cuda.is_available() and "device_map" not in kwargs:
                model = model.cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
            processor = None
            print("InternVL3 loaded successfully.")
            current_model_id = model_id
        else:
            backend = "qwen"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_id)
            tokenizer = None
            print("Qwen VLM model and processor loaded successfully.")
            current_model_id = model_id
    except Exception as e:
        print(f"ERROR: Failed to load VLM model. Error: {e}")
        # Fallback: if 8-bit quantization failed, retry without quantization
        model = None
        processor = None
        tokenizer = None


def unload_vlm_model():
    """Unload any loaded VLM and clear CUDA cache to free memory/defragment."""
    global model, processor, tokenizer, backend, current_model_id
    try:
        del model
    except Exception:
        pass
    try:
        if processor is not None:
            del processor
    except Exception:
        pass
    try:
        if tokenizer is not None:
            del tokenizer
    except Exception:
        pass
    model = None
    processor = None
    tokenizer = None
    backend = None
    current_model_id = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    print("Unloaded VLM model and cleared GPU cache")


def read_activities_context(yaml_path: str, activity_name: str) -> Dict[str, Any]:
    if not os.path.exists(yaml_path):
        return {"name": activity_name, "workspace": None, "target_objects": None, "steps": []}
    with open(yaml_path, 'r') as yf:
        data = yaml.safe_load(yf)
    name_norm = str(activity_name).strip().lower()
    for item in data:
        if str(item.get('name', '')).strip().lower() == name_norm:
            return {
                "name": item.get('name'),
                "workspace": item.get('workspace'),
                "target_objects": item.get('target_objects'),
                "steps": item.get('steps', [])
            }
    return {"name": activity_name, "workspace": None, "target_objects": None, "steps": []}


def infer_analysis_fps(motion_df: pd.DataFrame, hint_fps: float = None) -> float:
    if hint_fps and hint_fps > 0:
        return float(hint_fps)
    if 'time_s' in motion_df.columns and len(motion_df) > 1:
        dt = float(pd.Series(motion_df['time_s'].values).diff().median())
        if dt > 0:
            return float(1.0 / dt)
    return 10.0


def build_windows(n_frames: int, fps: float, window_s: float, overlap: float) -> List[Tuple[int, int, float, float]]:
    if fps <= 0:
        fps = 10.0
    step = max(1, int(round(window_s * fps * (1.0 - overlap))))
    win_len = max(1, int(round(window_s * fps)))
    windows = []
    start = 0
    while start < n_frames:
        end = min(n_frames, start + win_len)
        start_t = start / fps
        end_t = (end - 1) / fps if end > start else start_t
        windows.append((start, end, start_t, end_t))
        if end >= n_frames:
            break
        start += step
    return windows


def get_frame_at_time(cap: cv2.VideoCapture, t_s: float, fps: float, width: int, height: int) -> np.ndarray:
    if fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = max(0, int(round(t_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    return frame


def build_vlm_system_prompt() -> str:
    return (
        "You are a precise vision-language assistant for rehabilitation movement analysis. "
        "Multiple people may appear in the frames. Focus ONLY on the PATIENT specified by the provided keypoints JSON. "
        "IMPORTANT: Determine contact based on the patient's HAND/FINGERS/PALM that correspond to the wrist keypoint; do NOT judge contact by the wrist joint alone. "
        "Consider the temporal context and activity phase when consistent with the provided recent window history. "
        "If ANY frame in the provided window shows contact, classify the ENTIRE window as contact=1. "
        "Respond ONLY with a compact JSON object as described. "
        "Always include a numeric confidence between 0.05 and 0.95 (two decimals)."
    )


def build_vlm_user_json_payload(activity_ctx: Dict[str, Any], hand: str, window_meta: Dict[str, Any], keypoints: Dict[str, Any],
                                 recent_history: List[Dict[str, Any]], temporal_prior: str, current_step_hint: str = None) -> Dict[str, Any]:
    return {
        "task": "binary_contact_detection",
        "instructions": {
            "what_to_decide": "Return contact=1 if in ANY frame in this window the patient's hand/fingers/palm (associated with the given wrist keypoint) physically contacts the target object(s). If no frames show contact, return contact=0.",
            "hand": hand,
            "definitions": {
                "contact": "Any visible touching, grasping, holding between the target hand/fingers/palm and the object(s) (hand where the wrist, elbow, and shoulder keypoints are annotated). Do not consider the wrist alone as contact.",
                "target_objects": "Objects listed for the current activity context (from the dataset protocol).",
                "temporal_prior": temporal_prior
            },
            "output_format": {
                "type": "json",
                "schema": {"contact": "0 or 1 integer", "confidence": "0-1 float", "rationale": "brief string (<= 200 chars)"}
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
        "keypoints": keypoints
    }


def draw_keypoints_on_frame(frame: np.ndarray, kp: Dict[str, Any]) -> np.ndarray:
    """Draw keypoints and connections (used only for saved debug videos, not VLM input)."""
    # Extract keypoint coordinates
    shoulder_x, shoulder_y = kp["shoulder"]["x"], kp["shoulder"]["y"]
    elbow_x, elbow_y = kp["elbow"]["x"], kp["elbow"]["y"]
    wrist_x, wrist_y = kp["wrist"]["x"], kp["wrist"]["y"]
    
    # Only draw if we have valid coordinates
    if not (math.isnan(shoulder_x) or math.isnan(shoulder_y)):
        shoulder_pt = (int(shoulder_x), int(shoulder_y))
        elbow_pt = (int(elbow_x), int(elbow_y)) if not math.isnan(elbow_x) else None
        wrist_pt = (int(wrist_x), int(wrist_y)) if not math.isnan(wrist_x) else None
        
        # Draw keypoints with larger, more visible circles
        cv2.circle(frame, shoulder_pt, 12, (255, 0, 0), -1)  # Blue shoulder
        cv2.circle(frame, shoulder_pt, 15, (255, 255, 255), 3)  # White border
        
        if elbow_pt:
            cv2.circle(frame, elbow_pt, 12, (0, 255, 0), -1)  # Green elbow
            cv2.circle(frame, elbow_pt, 15, (255, 255, 255), 3)  # White border
            
        if wrist_pt:
            cv2.circle(frame, wrist_pt, 12, (0, 0, 255), -1)  # Red wrist
            cv2.circle(frame, wrist_pt, 15, (255, 255, 255), 3)  # White border
            
            # Add hand region indicator - approximate hand area beyond wrist
            # Estimate hand direction from elbow to wrist
            if elbow_pt:
                # Calculate direction vector from elbow to wrist
                dx = wrist_pt[0] - elbow_pt[0]
                dy = wrist_pt[1] - elbow_pt[1]
                # Normalize and extend to approximate hand length (about 20% of forearm)
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    hand_length = int(length * 0.3)  # Hand is roughly 30% of forearm length
                    hand_dx = int((dx / length) * hand_length)
                    hand_dy = int((dy / length) * hand_length)
                    hand_end_pt = (wrist_pt[0] + hand_dx, wrist_pt[1] + hand_dy)
                    
                    # Draw hand region as a semi-transparent circle
                    overlay = frame.copy()
                    cv2.circle(overlay, hand_end_pt, 25, (255, 0, 255), -1)  # Magenta hand area
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
                    # Draw hand direction line
                    cv2.line(frame, wrist_pt, hand_end_pt, (255, 0, 255), 4)  # Magenta hand line
                    cv2.circle(frame, hand_end_pt, 25, (255, 0, 255), 3)  # Hand area outline
        
        # Draw connections (arm structure) with thick lines
        if elbow_pt:
            cv2.line(frame, shoulder_pt, elbow_pt, (255, 255, 0), 6)  # Thick yellow line
        if wrist_pt and elbow_pt:
            cv2.line(frame, elbow_pt, wrist_pt, (255, 255, 0), 6)  # Thick yellow line
        
        # Add clear labels with background
        def add_label_with_bg(frame, text, pos, color=(255, 255, 255)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            # Black background rectangle
            cv2.rectangle(frame, (pos[0]-5, pos[1]-text_size[1]-5), 
                         (pos[0]+text_size[0]+5, pos[1]+5), (0, 0, 0), -1)
            # White text
            cv2.putText(frame, text, pos, font, font_scale, color, thickness)
        
        add_label_with_bg(frame, 'SHOULDER', (shoulder_pt[0]-20, shoulder_pt[1]-20))
        if elbow_pt:
            add_label_with_bg(frame, 'ELBOW', (elbow_pt[0]-20, elbow_pt[1]-20))
        if wrist_pt:
            add_label_with_bg(frame, 'WRIST', (wrist_pt[0]-20, wrist_pt[1]-20))
            # Add hand area label if we have the hand region
            # if elbow_pt:
            #     dx = wrist_pt[0] - elbow_pt[0]
            #     dy = wrist_pt[1] - elbow_pt[1]
            #     length = math.sqrt(dx*dx + dy*dy)
            #     if length > 0:
            #         hand_length = int(length * 0.3)
            #         hand_dx = int((dx / length) * hand_length)
            #         hand_dy = int((dy / length) * hand_length)
            #         hand_end_pt = (wrist_pt[0] + hand_dx, wrist_pt[1] + hand_dy)
            #         add_label_with_bg(frame, 'HAND AREA', (hand_end_pt[0]-30, hand_end_pt[1]-30), (255, 0, 255))
        
        # Add a prominent text overlay indicating this is the PATIENT
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 255, 255), 3)  # Yellow border
        cv2.putText(frame, 'PATIENT (with sensors)', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, 'Look for HAND/FINGER contact', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return frame


def write_window_video(video_path: str, start_t: float, end_t: float, 
                      fps: float, width: int, height: int, output_path: str,
                      contact: int, confidence: float, rationale: str,
                      motion_df: pd.DataFrame, window_start_idx: int, window_end_idx: int, handedness: str):
    """Write a video segment with contact information overlay and highlighted keypoints"""
    # Create a separate capture to avoid interfering with main capture
    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        print(f"Warning: Could not open video capture for writing window video {output_path}")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Warning: Could not open video writer for {output_path}")
        cap2.release()
        return
    
    # Calculate frame range
    start_frame = int(start_t * fps)
    end_frame = int(end_t * fps)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Prepare text overlay
    text_lines = [
        f"Contact: {'YES' if contact else 'NO'}",
        f"Confidence: {confidence:.2f}",
        "Rationale:"
    ] + textwrap.wrap(rationale, width=40)  # Wrap long rationale
    
    # Process each frame in the window
    frame_idx = 0
    for _ in range(start_frame, end_frame):
        ret, frame = cap2.read()
        if not ret:
            break
        
        # Calculate corresponding motion data index
        motion_idx = window_start_idx + int((frame_idx / max(1, end_frame - start_frame)) * (window_end_idx - window_start_idx))
        motion_idx = min(motion_idx, len(motion_df) - 1)
        
        # Get keypoints for this frame and highlight them
        if motion_idx < len(motion_df):
            row = motion_df.iloc[motion_idx]
            kp = {
                "shoulder": {"x": float(row.get('shoulder_x', np.nan)), "y": float(row.get('shoulder_y', np.nan))},
                "elbow": {"x": float(row.get('elbow_x', np.nan)), "y": float(row.get('elbow_y', np.nan))},
                "wrist": {"x": float(row.get('wrist_x', np.nan)), "y": float(row.get('wrist_y', np.nan))},
                "overall_confidence": float(row.get('overall_confidence', 0.0))
            }
            # Fallback to handedness-specific columns if generic missing
            if math.isnan(kp['shoulder']['x']) or math.isnan(kp['shoulder']['y']):
                hand_pref = 'left' if handedness == 'L' else 'right'
                kp = {
                    "shoulder": {"x": float(row.get(f'{hand_pref}_shoulder_x', np.nan)), "y": float(row.get(f'{hand_pref}_shoulder_y', np.nan))},
                    "elbow": {"x": float(row.get(f'{hand_pref}_elbow_x', np.nan)), "y": float(row.get(f'{hand_pref}_elbow_y', np.nan))},
                    "wrist": {"x": float(row.get(f'{hand_pref}_wrist_x', np.nan)), "y": float(row.get(f'{hand_pref}_wrist_y', np.nan))},
                    "overall_confidence": float(row.get('overall_confidence', 0.0))
                }
            
            # Highlight keypoints on this frame
            frame = draw_keypoints_on_frame(frame, kp)
            
        # Add text overlay with semi-transparent background
        overlay = frame.copy()
        y_start = height - len(text_lines) * 35 - 20  # Position at bottom
        cv2.rectangle(overlay, (10, y_start - 10), (450, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # Semi-transparent background
        
        for i, line in enumerate(text_lines):
            y = y_start + i * 30
            cv2.putText(
                frame, line, (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 0, 255) if contact else (0, 255, 0), 
                2
            )
            
        out.write(frame)
        frame_idx += 1
        
    out.release()
    cap2.release()


def get_window_frames_pil(video_path: str, start_t: float, end_t: float, fps_video: float,
                         width: int, height: int, max_frames: int = 16, playback_speed: float = 1.0) -> List[Image.Image]:
    """Sample raw frames across a window and return PIL images for VLM video input (no overlays)."""
    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        return []
    times = np.linspace(start_t, end_t, max(2, max_frames)).tolist()
    pil_frames: List[Image.Image] = []
    for t in times:
        frame = get_frame_at_time(cap2, float(t), fps_video, width, height)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Do not duplicate frames based on playback_speed; keep prompt short and bounded by max_frames
        pil_frames.append(img)
    cap2.release()
    return pil_frames


def call_vlm_contact_video(video_frames_pil: List[Image.Image], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Performs local inference using Qwen2.5-VL on a short video (sequence of frames)."""
    global model, processor, backend
    if model is None or (backend == "qwen" and processor is None):
        return {"contact": 0, "confidence": 0.0, "rationale": "model_not_loaded"}

    system_prompt = build_vlm_system_prompt()
    user_json_text = json.dumps(payload, ensure_ascii=False)

    try:
        if backend == "internvl3":
            # Prepend a strong system instruction to the user text for InternVL3
            content = internvl3_chat_video(video_frames_pil, system_prompt + "\n\n" + user_json_text + "\nRespond ONLY with the JSON.")
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "video", "video": video_frames_pil},
                    {"type": "text", "text": user_json_text},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=250, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            content = output_text.strip()
        print(f"VLM raw response: {content}")
    except Exception as e:
        return {"contact": 0, "confidence": 0.0, "rationale": f"inference_error: {str(e)}"}

    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start == -1 or end == -1:
            raise json.JSONDecodeError("No JSON object found", content, 0)
        parsed = json.loads(content[start:end])
        contact = int(parsed.get("contact", 0))
        conf = float(parsed.get("confidence", 0.0))
        # Clamp confidence to a reasonable open interval to avoid exact 0.0 causing downstream zeros
        conf = max(0.05, min(0.95, conf))
        rationale = str(parsed.get("rationale", ""))
        return {"contact": 1 if contact else 0, "confidence": max(0.0, min(1.0, conf)), "rationale": rationale[:200]}
    except Exception as e:
        return {"contact": 0, "confidence": 0.0, "rationale": f"parse_error: {content}"}


def call_vlm_contact(image_bgr: np.ndarray, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Backend-aware single-image contact call for Qwen2.5-VL or InternVL3.

    For InternVL3, uses tiling + chat(image) path. For Qwen, uses processor-based path.
    """
    global model, processor, backend
    if model is None:
        return {"contact": 0, "confidence": 0.0, "rationale": "model_not_loaded"}

    system_prompt = build_vlm_system_prompt()
    user_json_text = json.dumps(payload, ensure_ascii=False)

    # Convert image from OpenCV BGR to PIL RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    try:
        if backend == "internvl3":
            # Use InternVL3 image chat with explicit system prompt injected
            content = internvl3_chat_image(pil_image, system_prompt + "\n\n" + user_json_text + "\nRespond ONLY with the JSON.")
        else:
            # Qwen official path
            if processor is None:
                return {"contact": 0, "confidence": 0.0, "rationale": "processor_not_loaded"}
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_json_text},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=250, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            content = output_text.strip()
        print(f"VLM raw response: {content}")
    except Exception as e:
        return {"contact": 0, "confidence": 0.0, "rationale": f"inference_error: {str(e)}"}

    # Parse JSON
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start == -1 or end == -1:
            raise json.JSONDecodeError("No JSON object found", content, 0)
        parsed = json.loads(content[start:end])
        contact = int(parsed.get("contact", 0))
        conf = float(parsed.get("confidence", 0.0))
        conf = max(0.05, min(0.95, conf))
        rationale = str(parsed.get("rationale", ""))
        return {"contact": 1 if contact else 0, "confidence": max(0.0, min(1.0, conf)), "rationale": rationale[:200]}
    except Exception:
        return {"contact": 0, "confidence": 0.0, "rationale": f"parse_error: {content}"}


def call_vlm_contact_batch(images_bgr: List[np.ndarray], payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batched contact inference for Qwen; falls back to per-image for InternVL3 or on error.

    Each payload corresponds to the matching image.
    """
    global model, processor, backend
    results: List[Dict[str, Any]] = []
    if model is None:
        return [{"contact": 0, "confidence": 0.0, "rationale": "model_not_loaded"} for _ in images_bgr]

    # InternVL3 path: no simple batch path with current chat() API; process sequentially
    if backend == "internvl3":
        for img, pl in zip(images_bgr, payloads):
            results.append(call_vlm_contact(img, pl))
        return results

    # Qwen path with processor batching
    if processor is None:
        return [{"contact": 0, "confidence": 0.0, "rationale": "processor_not_loaded"} for _ in images_bgr]

    try:
        system_prompt = build_vlm_system_prompt()
        messages_list = []
        for img_bgr, pl in zip(images_bgr, payloads):
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            user_json_text = json.dumps(pl, ensure_ascii=False)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_json_text},
                ]},
            ]
            messages_list.append(messages)

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        image_inputs_list, video_inputs_list = [], []
        for m in messages_list:
            image_inputs, video_inputs = process_vision_info(m)
            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)

        inputs = processor(
            text=texts,
            images=image_inputs_list,
            videos=video_inputs_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=250, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for content in outputs:
            content = str(content).strip()
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start == -1 or end == -1:
                    raise json.JSONDecodeError("No JSON object found", content, 0)
                parsed = json.loads(content[start:end])
                contact = int(parsed.get("contact", 0))
                conf = float(parsed.get("confidence", 0.0))
                conf = max(0.05, min(0.95, conf))
                rationale = str(parsed.get("rationale", ""))
                results.append({"contact": 1 if contact else 0, "confidence": max(0.0, min(1.0, conf)), "rationale": rationale[:200]})
            except Exception:
                results.append({"contact": 0, "confidence": 0.0, "rationale": f"parse_error: {content}"})
    except Exception:
        # Fallback to per-image on any batch error
        results = [call_vlm_contact(img, pl) for img, pl in zip(images_bgr, payloads)]

    return results

def build_temporal_prior(activity_ctx: Dict[str, Any]) -> str:
    """Create a brief temporal prior string derived from the activity context.
    E.g., for combing: expect pick up -> use -> put back, implying at most two contact transitions.
    """
    name = str(activity_ctx.get("name", "")).strip().lower()
    steps = [str(s).lower() for s in activity_ctx.get("steps", [])]

    # Generic prior default
    prior = (
        "Use temporal consistency: contact segments tend to be contiguous; "
        "expect a small number of transitions consistent with the activity steps."
    )

    if "comb" in name or any("comb" in s for s in steps):
        prior = (
            "Activity involves picking up a comb, using it, then putting it back. "
            "Expect two transitions overall: NO_CONTACT→CONTACT once (pick up), then CONTACT→NO_CONTACT (put back)."
        )
    elif "drink" in name or any("sip" in s or "pour" in s for s in steps):
        prior = (
            "Activity involves grasping a cup/bottle to drink and releasing after. "
            "Expect intermittent CONTACT during grasp/use and NO_CONTACT otherwise; transitions are few."
        )
    elif "glass" in name or any("glasses" in s for s in steps):
        prior = (
            "Activity involves donning and doffing glasses. "
            "Expect brief CONTACT near face when putting on/removing; otherwise NO_CONTACT."
        )
    elif "brush" in name or any("toothbrush" in s or "toothpaste" in s for s in steps):
        prior = (
            "Activity involves grasping a toothbrush for a sustained period while brushing, then releasing. "
            "Expect one CONTACT onset and one offset surrounding brushing."
        )
    elif "deodor" in name or any("deodorant" in s for s in steps):
        prior = (
            "Activity involves grasping a deodorant stick, using it, then releasing. "
            "Expect few transitions: NO_CONTACT→CONTACT, then CONTACT→NO_CONTACT."
        )
    else:
        # Fall back to steps to infer a simple pick/use/return pattern if present
        if any("pick" in s for s in steps) and any("place" in s or "put" in s for s in steps):
            prior = (
                "Activity suggests pick up, use, and put back. "
                "Expect two transitions: NO_CONTACT→CONTACT, then CONTACT→NO_CONTACT."
            )

    return prior


def internvl3_chat_image(pil_image: Image.Image, user_text: str) -> str:
    global model, tokenizer
    # Prepare tiles
    tiles = dynamic_preprocess(pil_image, image_size=448, use_thumbnail=True, max_num=12)
    transform = build_transform(448)
    pixel_values = [transform(tile) for tile in tiles]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    question = '<image>\n' + user_text
    generation_config = dict(max_new_tokens=512, do_sample=False)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return str(response)


def internvl3_chat_video(pil_frames: List[Image.Image], user_text: str) -> str:
    global model, tokenizer
    # Cap number of frames to avoid overlong prompts
    if len(pil_frames) > 8:
        pil_frames = pil_frames[:8]
    transform = build_transform(448)
    pixel_values_list, num_patches_list = [], []
    for frame in pil_frames:
        tiles = dynamic_preprocess(frame, image_size=448, use_thumbnail=True, max_num=1)
        pv = [transform(tile) for tile in tiles]
        pv = torch.stack(pv)
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)
    pixel_values = torch.cat(pixel_values_list)
    pixel_values = pixel_values.to(torch.bfloat16)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + user_text
    generation_config = dict(max_new_tokens=512, do_sample=False)
    response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                             num_patches_list=num_patches_list, history=None, return_history=True)
    return str(response)


def run_contact_detection(motion_csv, video_path, activities_yaml, activity, handedness="L", 
                         window_s=1.0, overlap=0.5, analysis_fps=0.0, model="Qwen/Qwen2.5-VL-7B-Instruct",
                         internvl_quant="none", internvl_split=False, max_frames=16, 
                         output_csv=None, window_videos_dir=None, playback_speed: float = 1.0):
    """
    Run VLM contact detection as a function call.
    
    Args:
        motion_csv: Path to enhanced motion data CSV
        video_path: Path to original video
        activities_yaml: Path to activities_ground_truth.yaml
        activity: Activity name for this video
        handedness: Target hand ("L" or "R")
        window_s: Window size in seconds
        overlap: Window overlap fraction (0-1)
        analysis_fps: Analysis FPS; if 0, inferred from motion CSV
        model: VLM model id
        internvl_quant: InternVL3 quantization ("none" or "8bit")
        internvl_split: Enable multi-GPU split for InternVL3
        max_frames: Max frames sampled per window for VLM video input
        output_csv: Path to output CSV with window results
        window_videos_dir: Directory to save window videos with overlay
    
    Returns:
        dict: Results containing CSV path and other outputs
    """
    print("🤖 Running VLM contact detection...")
    
    # Load the VLM model into memory at startup
    load_vlm_model(model, internvl_quant=internvl_quant, internvl_split=internvl_split)

    if not os.path.exists(motion_csv):
        raise FileNotFoundError(motion_csv)
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    motion_df = pd.read_csv(motion_csv)
    fps_analysis = infer_analysis_fps(motion_df, analysis_fps)

    n_frames = len(motion_df)
    windows = build_windows(n_frames, fps_analysis, window_s, overlap)

    # Prepare video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or fps_analysis)

    # Activity context
    activity_ctx = read_activities_context(activities_yaml, activity)

    # Create window videos directory if specified
    if window_videos_dir:
        os.makedirs(window_videos_dir, exist_ok=True)

    results = []
    steps_list = activity_ctx.get("steps", []) if isinstance(activity_ctx.get("steps", []), list) else []
    total_windows = max(1, len(windows))
    for w_i, (s_idx, e_idx, s_t, e_t) in enumerate(tqdm(windows, desc="VLM Contact Detection")):
        # Motion per window (majority vote)
        motion_col = 'prediction' if 'prediction' in motion_df.columns else ('predictions' if 'predictions' in motion_df.columns else None)
        if motion_col is None:
            motion_val = 1 if float(motion_df.get('probability', pd.Series(np.zeros(n_frames))).iloc[s_idx:e_idx].mean()) >= 0.5 else 0
        else:
            window_vals = motion_df[motion_col].iloc[s_idx:e_idx].values.astype(int)
            motion_val = int(window_vals.mean() >= 0.5)

        # Center frame time and keypoints at nearest index
        center_idx = (s_idx + e_idx) // 2
        center_t = (s_t + e_t) / 2.0
        frame = get_frame_at_time(cap, center_t, fps_video, width, height)

        # Keypoints snapshot
        row = motion_df.iloc[center_idx]
        kp = {
            "shoulder": {"x": float(row.get('shoulder_x', np.nan)), "y": float(row.get('shoulder_y', np.nan))},
            "elbow": {"x": float(row.get('elbow_x', np.nan)), "y": float(row.get('elbow_y', np.nan))},
            "wrist": {"x": float(row.get('wrist_x', np.nan)), "y": float(row.get('wrist_y', np.nan))},
            "overall_confidence": float(row.get('overall_confidence', 0.0))
        }
        # Fallback to handedness-specific columns if generic missing
        if math.isnan(kp['shoulder']['x']) or math.isnan(kp['shoulder']['y']):
            hand_pref = 'left' if handedness == 'L' else 'right'
            kp = {
                "shoulder": {"x": float(row.get(f'{hand_pref}_shoulder_x', np.nan)), "y": float(row.get(f'{hand_pref}_shoulder_y', np.nan))},
                "elbow": {"x": float(row.get(f'{hand_pref}_elbow_x', np.nan)), "y": float(row.get(f'{hand_pref}_elbow_y', np.nan))},
                "wrist": {"x": float(row.get(f'{hand_pref}_wrist_x', np.nan)), "y": float(row.get(f'{hand_pref}_wrist_y', np.nan))},
                "overall_confidence": float(row.get('overall_confidence', 0.0))
            }

        # Build temporal context and payload
        window_meta = {"start_time_s": float(s_t), "end_time_s": float(e_t), "duration_s": float(max(0.0, e_t - s_t))}
        recent_history = [{"start_time": r["start_time"], "end_time": r["end_time"], "motion": r["motion"], "contact": r.get("contact", None)} for r in results[-3:]]
        temporal_prior = build_temporal_prior(activity_ctx)
        # Current-step hint by proportional mapping
        cur_step_hint = None
        if steps_list:
            step_idx = int(round((w_i / max(1, total_windows - 1)) * (len(steps_list) - 1)))
            step_idx = min(max(0, step_idx), len(steps_list) - 1)
            cur_step_hint = str(steps_list[step_idx])
        payload = build_vlm_user_json_payload(activity_ctx, handedness, window_meta, kp, recent_history, temporal_prior, current_step_hint=cur_step_hint)

        # Prefer sending a short raw clip (video) across the window to the VLM
        video_frames_pil = get_window_frames_pil(
            video_path, s_t, e_t, fps_video, width, height,
            max_frames=max(2, int(max_frames)), playback_speed=float(playback_speed)
        )
        if len(video_frames_pil) >= 2:
            vlm_out = call_vlm_contact_video(video_frames_pil, payload)
        else:
            vlm_out = call_vlm_contact(frame, payload)
        contact = int(vlm_out.get('contact', 0))
        confidence = float(vlm_out.get('confidence', 0.0))
        rationale = str(vlm_out.get('rationale', ''))

        # Save window video with overlay if requested (keep overlays for human review only)
        if window_videos_dir:
            video_filename = os.path.join(
                window_videos_dir, 
                f"window_{s_t:.1f}s_to_{e_t:.1f}s.mp4"
            )
            write_window_video(
                video_path, s_t, e_t, fps_video, width, height,
                video_filename, contact, confidence, rationale,
                motion_df, s_idx, e_idx, handedness
            )

        results.append({
            "start_time": float(s_t),
            "end_time": float(e_t),
            "motion": int(motion_val),
            "contact": int(contact),
            "confidence": confidence,
            "rationale": rationale
        })

    cap.release()

    # Generate output CSV path if not provided
    if output_csv is None:
        video_identifier = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = f"{video_identifier}_window_contact.csv"

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df = pd.DataFrame(results, columns=["start_time", "end_time", "motion", "contact", "confidence", "rationale"])
    out_df.to_csv(output_csv, index=False)
    print(f"Window contact CSV saved to {output_csv}")
    
    return {
        "contact_csv": output_csv,
        "window_videos_dir": window_videos_dir,
        "total_windows": len(results)
    }


def run_contact_detection_framewise(motion_csv: str, video_path: str, activities_yaml: str, activity: str,
                                   handedness: str = "L", frame_fps: float = 60.0,
                                   gaussian_sigma: float = 1.0,
                                   high_threshold: float = 0.7,
                                   low_threshold: float = 0.3,
                                   model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                                   output_csv: str = None, window_videos_dir: str = None,
                                   batch_size: int = 8,
                                   median_kernel: int = 3,
                                   min_run_frames: int = 3,
                                   gap_fill_frames: int = 2) -> Dict[str, Any]:
    """Framewise contact classification using VLM on each sampled frame, Gaussian smoothing, and hysteresis.

    Produces a window-style CSV by collapsing consecutive frames with the same contact prediction.
    """
    print("🤖 Running VLM contact detection (framewise mode)...")

    load_vlm_model(model)

    if not os.path.exists(motion_csv):
        raise FileNotFoundError(motion_csv)
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    motion_df = pd.read_csv(motion_csv)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    # Sampling times at desired frame_fps over the video duration
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / max(1e-6, fps_video)
    times = np.arange(0.0, duration_s, 1.0 / max(1e-6, frame_fps))

    # Activity context (unused in per-frame prompt, but available if needed)
    activity_ctx = read_activities_context(activities_yaml, activity)
    temporal_prior = build_temporal_prior(activity_ctx)

    probs: List[float] = []

    # Build per-frame payloads and frames for batching
    kps: List[Dict[str, Any]] = []
    payloads: List[Dict[str, Any]] = []
    frames_bgr: List[np.ndarray] = []
    for t in times:
        if 'time_s' in motion_df.columns:
            nearest_idx = int(np.argmin(np.abs(motion_df['time_s'].values - t)))
        else:
            nearest_idx = min(len(motion_df) - 1, int(round(t * max(1.0, frame_fps))))
        row = motion_df.iloc[nearest_idx]
        kp = {
            "shoulder": {"x": float(row.get('shoulder_x', np.nan)), "y": float(row.get('shoulder_y', np.nan))},
            "elbow": {"x": float(row.get('elbow_x', np.nan)), "y": float(row.get('elbow_y', np.nan))},
            "wrist": {"x": float(row.get('wrist_x', np.nan)), "y": float(row.get('wrist_y', np.nan))},
            "overall_confidence": float(row.get('overall_confidence', 0.0))
        }
        if math.isnan(kp['shoulder']['x']) or math.isnan(kp['shoulder']['y']):
            hand_pref = 'left' if handedness == 'L' else 'right'
            kp = {
                "shoulder": {"x": float(row.get(f'{hand_pref}_shoulder_x', np.nan)), "y": float(row.get(f'{hand_pref}_shoulder_y', np.nan))},
                "elbow": {"x": float(row.get(f'{hand_pref}_elbow_x', np.nan)), "y": float(row.get(f'{hand_pref}_elbow_y', np.nan))},
                "wrist": {"x": float(row.get(f'{hand_pref}_wrist_x', np.nan)), "y": float(row.get(f'{hand_pref}_wrist_y', np.nan))},
                "overall_confidence": float(row.get('overall_confidence', 0.0))
            }
        window_meta = {"start_time_s": float(t), "end_time_s": float(t), "duration_s": 0.0}
        payloads.append(build_vlm_user_json_payload(activity_ctx, handedness, window_meta, kp, recent_history=[], temporal_prior=temporal_prior))
        frames_bgr.append(get_frame_at_time(cap, float(t), fps_video, width, height))

    for start in tqdm(range(0, len(times), max(1, int(batch_size))), desc="VLM Contact (framewise batched)"):
        end = min(len(times), start + max(1, int(batch_size)))
        batch_frames = frames_bgr[start:end]
        batch_payloads = payloads[start:end]
        outs = call_vlm_contact_batch(batch_frames, batch_payloads)
        for out in outs:
            conf = float(out.get('confidence', 0.0))
            contact_bin = int(out.get('contact', 0))
            p = conf if contact_bin == 1 else (1.0 - conf)
            probs.append(max(0.0, min(1.0, p)))

    cap.release()

    # Median filter + Gaussian smoothing
    if len(probs) == 0:
        probs = [0.0]
    prob_arr = np.array(probs, dtype=float)
    if int(median_kernel) >= 3 and int(median_kernel) % 2 == 1:
        prob_arr = median_filter(prob_arr, size=int(median_kernel), mode='nearest')
    smoothed = gaussian_filter1d(prob_arr, sigma=float(max(0.0, gaussian_sigma)))

    # Hysteresis thresholding
    contact_state = 0
    contact_series: List[int] = []
    hi = float(high_threshold)
    lo = float(low_threshold)
    for val in smoothed:
        if contact_state == 0 and val >= hi:
            contact_state = 1
        elif contact_state == 1 and val <= lo:
            contact_state = 0
        contact_series.append(contact_state)

    # Gap fill small 0-gaps between 1s
    if int(gap_fill_frames) > 0 and len(contact_series) > 0:
        i = 0
        while i < len(contact_series):
            if contact_series[i] == 1:
                j = i + 1
                while j < len(contact_series) and contact_series[j] == 1:
                    j += 1
                k = j
                while k < len(contact_series) and contact_series[k] == 0 and (k - j) <= int(gap_fill_frames):
                    k += 1
                if k < len(contact_series) and contact_series[k] == 1 and (k - j) <= int(gap_fill_frames):
                    contact_series[j:k] = 1
                i = k
            else:
                i += 1

    # Enforce minimum run length for contact segments
    if int(min_run_frames) > 1 and len(contact_series) > 0:
        i = 0
        while i < len(contact_series):
            if contact_series[i] == 1:
                j = i
                while j < len(contact_series) and contact_series[j] == 1:
                    j += 1
                if (j - i) < int(min_run_frames):
                    contact_series[i:j] = 0
                i = j
            else:
                i += 1

    # Collapse runs of equal contact into windows
    results = []
    if len(times) > 0:
        start_t = times[0]
        cur_state = contact_series[0]
        for i in range(1, len(times)):
            if contact_series[i] != cur_state:
                end_t = times[i]
                conf_slice = smoothed[max(0, i-5):i+1]
                results.append({
                    "start_time": float(start_t),
                    "end_time": float(end_t),
                    "motion": int(1),
                    "contact": int(cur_state),
                    "confidence": float(np.mean(conf_slice if conf_slice.size > 0 else smoothed)),
                    "rationale": "framewise_gaussian_hysteresis_median_minrun_gapfill"
                })
                start_t = times[i]
                cur_state = contact_series[i]
        conf_tail = smoothed[-5:] if len(smoothed) >= 5 else smoothed
        results.append({
            "start_time": float(start_t),
            "end_time": float(times[-1] if len(times) > 0 else start_t),
            "motion": int(1),
            "contact": int(cur_state),
            "confidence": float(np.mean(conf_tail)),
            "rationale": "framewise_gaussian_hysteresis_median_minrun_gapfill"
        })

    if output_csv is None:
        video_identifier = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = f"{video_identifier}_framewise_contact.csv"

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df = pd.DataFrame(results, columns=["start_time", "end_time", "motion", "contact", "confidence", "rationale"])
    out_df.to_csv(output_csv, index=False)
    print(f"Framewise contact CSV saved to {output_csv}")

    return {
        "contact_csv": output_csv,
        "window_videos_dir": window_videos_dir,
        "total_windows": len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="Windowed VLM contact detection for rehab videos")
    parser.add_argument("--motion_csv", type=str, required=True, help="Path to enhanced motion data CSV")
    parser.add_argument("--video_path", type=str, required=True, help="Path to original video")
    parser.add_argument("--activities_yaml", type=str, required=True, help="Path to activities_ground_truth.yaml")
    parser.add_argument("--activity", type=str, required=True, help="Activity name for this video")
    parser.add_argument("--handedness", type=str, default="L", choices=["L","R"], help="Target hand")
    parser.add_argument("--window_s", type=float, default=1.0, help="Window size in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction (0-1)")
    parser.add_argument("--analysis_fps", type=float, default=0.0, help="Analysis FPS; if 0, inferred from motion CSV")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model id")
    parser.add_argument("--internvl_quant", type=str, default="none", choices=["none","8bit"], help="InternVL3 quantization (if using InternVL3)")
    parser.add_argument("--internvl_split", action="store_true", help="Enable multi-GPU split for InternVL3 (distributes LLM layers)")
    parser.add_argument("--max_frames", type=int, default=16, help="Max frames sampled per window for VLM video input")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV with window results")
    parser.add_argument("--window_videos_dir", type=str, default=None, help="Directory to save window videos with overlay. If not provided, no videos are saved.")
    args = parser.parse_args()

    # Call the function version
    results = run_contact_detection(
        motion_csv=args.motion_csv,
        video_path=args.video_path,
        activities_yaml=args.activities_yaml,
        activity=args.activity,
        handedness=args.handedness,
        window_s=args.window_s,
        overlap=args.overlap,
        analysis_fps=args.analysis_fps,
        model=args.model,
        internvl_quant=args.internvl_quant,
        internvl_split=args.internvl_split,
        max_frames=args.max_frames,
        output_csv=args.output_csv,
        window_videos_dir=args.window_videos_dir
    )
    
    return results


if __name__ == "__main__":
    main() 