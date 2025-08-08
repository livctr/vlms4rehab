import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration # Ensure correct model class
import os
from typing import Dict, Optional, Any, List

# --- Configuration (can be overridden by function parameters) ---
DEFAULT_LLM_MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_OBJECTS_TO_TRACK_PROMPT = "a fork, a knife, bread, or butter"
DEFAULT_FRAME_STRIDE = 1
DEFAULT_WRIST_ROI_PADDING = 50
DEFAULT_MIN_WRIST_CONFIDENCE_FOR_ROI = 0.3

SAVE_DEBUG_LLM_FRAMES = True
DEBUG_FRAME_SAVE_INTERVAL = 500

def llm_interaction_analyzer(
    selected_patient_id: int,
    hand_to_track: str, 
    video_path: str,
    mapped_hand_data_path: str = "mapped_hand_data_enhanced.json",
    output_video_path: str = "llm_interactions_annotated_wrist_roi.mp4",
    output_roi_details_json: str = "llm_roi_details_wrist_roi.json",
    output_interactions_json: str = "llm_interactions_log_wrist_roi.json",
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    objects_for_prompt: str = DEFAULT_OBJECTS_TO_TRACK_PROMPT,
    wrist_roi_padding: int = DEFAULT_WRIST_ROI_PADDING,
    min_wrist_confidence_for_roi: float = DEFAULT_MIN_WRIST_CONFIDENCE_FOR_ROI,
    save_debug_frames: bool = SAVE_DEBUG_LLM_FRAMES,
    debug_frame_interval: int = DEBUG_FRAME_SAVE_INTERVAL
):
    print(f"\n--- Starting LLM Interaction Analysis (ROI around Wrist) ---")
    print(f"Patient ID: {selected_patient_id}, Hand: {hand_to_track.upper()}")
    print(f"Video: {video_path}")
    print(f"Mapped Hand Data (for wrist coords): {mapped_hand_data_path}")
    print(f"LLM Model: {llm_model_name}")
    print(f"Wrist ROI Padding: {wrist_roi_padding}, Min Wrist Conf: {min_wrist_confidence_for_roi}")

    if hand_to_track.upper() not in ["L", "R"]:
        print("Error: hand_to_track must be 'L' or 'R'.")
        return False

    # === Load LLaVA model ===
    print("Loading LLaVA model...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = torch.float16 if device.type == 'cuda' else torch.float32
        
        processor = AutoProcessor.from_pretrained(llm_model_name)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            llm_model_name,
            torch_dtype=target_dtype, # Use target_dtype
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        # If not using device_map="auto" or to be absolutely sure for single GPU:
        # model.to(device) 
        model.eval()
        print(f"LLaVA model '{llm_model_name}' loaded successfully on {device} with dtype {model.dtype}.")
    except Exception as e:
        print(f"Error loading LLaVA model '{llm_model_name}': {e}"); return False

    # === Load mapped hand data (which contains wrist coordinates) ===
    print(f"Loading mapped hand data from: {mapped_hand_data_path}")
    try:
        with open(mapped_hand_data_path, "r") as f:
            mapped_data_all_frames = json.load(f)
    except Exception as e: print(f"Error loading mapped data: {e}"); return False

    # === Video Setup ===
    print(f"Setting up video processing for: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: Input video file not found: {video_path}"); return False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: Could not open video file {video_path}"); return False
    fps = cap.get(cv2.CAP_PROP_FPS); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video_dir = os.path.dirname(output_video_path)
    if output_video_dir and not os.path.exists(output_video_dir): os.makedirs(output_video_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    if not out_video_writer.isOpened():
        print(f"Error opening video writer for {output_video_path}"); cap.release(); return False
    print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {W}x{H}")

    # --- Processing ---
    llm_roi_details: Dict[str, Optional[Dict[str, Any]]] = {}
    llm_interactions: Dict[str, bool] = {}
    llm_query_count = 0

    debug_frame_output_dir = "llava_debug_frames_wrist_roi"
    if save_debug_frames: os.makedirs(debug_frame_output_dir, exist_ok=True)

    sorted_frame_keys = sorted(mapped_data_all_frames.keys(), key=int)

    print("Analyzing frames with LLaVA (ROI around wrist)...")
    for frame_idx_str in tqdm(sorted_frame_keys, desc="LLM Processing (Wrist ROI)"):
        current_frame_num = int(frame_idx_str)

        if current_frame_num % frame_stride != 0:
            continue
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) != current_frame_num:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
        
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {current_frame_num}. Skipping.")
            continue

        frame_rgb_for_llava_processing = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # For LLaVA
        annotated_frame_for_video = frame_bgr.copy() # For output video

        frame_specific_mapped_data = mapped_data_all_frames.get(frame_idx_str)
        roi_box_around_wrist: Optional[List[int]] = None 
        target_wrist_pt: Optional[List[float]] = None
        wrist_conf: Optional[float] = None
        hand_roi_color = (255, 0, 0) 

        if (frame_specific_mapped_data and
            frame_specific_mapped_data.get("selected_patient_id") == selected_patient_id and
            frame_specific_mapped_data.get("tracked_hand_label") == hand_to_track.upper()):
            
            target_wrist_pt = frame_specific_mapped_data.get("target_wrist_coords")
            wrist_conf = frame_specific_mapped_data.get("target_wrist_confidence")

            if target_wrist_pt and wrist_conf is not None and wrist_conf >= min_wrist_confidence_for_roi:
                cx, cy = int(target_wrist_pt[0]), int(target_wrist_pt[1])
                x1 = max(0, cx - wrist_roi_padding)
                y1 = max(0, cy - wrist_roi_padding)
                x2 = min(W, cx + wrist_roi_padding)
                y2 = min(H, cy + wrist_roi_padding)
                if x1 < x2 and y1 < y2: 
                    roi_box_around_wrist = [x1, y1, x2, y2]
            
            llm_roi_details[frame_idx_str] = {
                "target_wrist_coords": target_wrist_pt,
                "target_wrist_confidence": wrist_conf,
                "roi_box_around_wrist": roi_box_around_wrist 
            }
        else:
            llm_roi_details[frame_idx_str] = None

        llm_answer_display = "N/A (No ROI)"
        llm_interaction_result = False

        if roi_box_around_wrist: 
            # Draw the ROI box on the RGB frame that will be sent to LLaVA
            x1r, y1r, x2r, y2r = roi_box_around_wrist
            # Create a copy for LLaVA to avoid altering original if needed elsewhere
            frame_rgb_with_roi_for_llava = frame_rgb_for_llava_processing.copy() 
            cv2.rectangle(frame_rgb_with_roi_for_llava, (x1r, y1r), (x2r, y2r), hand_roi_color, 3) 
            
            # Also draw on BGR for output video later
            cv2.rectangle(annotated_frame_for_video, (x1r, y1r), (x2r, y2r), hand_roi_color, 2)
            cv2.putText(annotated_frame_for_video, f"{hand_to_track.upper()}-Hand ROI", (x1r, y1r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_roi_color, 2)

            llm_input_image_pil = Image.fromarray(frame_rgb_with_roi_for_llava) 
            
            if save_debug_frames and llm_query_count % debug_frame_interval == 0:
                Image.fromarray(frame_rgb_with_roi_for_llava).save(
                    os.path.join(debug_frame_output_dir, f"frame_{frame_idx_str}_llava_input_with_roi.png")
                )
            llm_query_count +=1
            
            prompt_text = (
                f"An image is provided. Focus on the hand within the red bounding box. "
                f"Is this hand actively grasping, holding, or clearly touching any of these: {objects_for_prompt}? "
                f"Respond with only the word \"TRUE\" or \"FALSE\"."
            )

            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            templated_input_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            final_inputs = processor(text=templated_input_text, images=llm_input_image_pil, return_tensors="pt").to(device)

            # **** KEY FIX: Ensure pixel_values matches model dtype ****
            if model.dtype == torch.float16 and 'pixel_values' in final_inputs and final_inputs['pixel_values'].dtype == torch.float32:
                final_inputs['pixel_values'] = final_inputs['pixel_values'].to(torch.float16)
            # Handle pixel_values_videos if your model version uses it
            if model.dtype == torch.float16 and 'pixel_values_videos' in final_inputs and \
               final_inputs.get('pixel_values_videos') is not None and \
               final_inputs['pixel_values_videos'].dtype == torch.float32:
                final_inputs['pixel_values_videos'] = final_inputs['pixel_values_videos'].to(torch.float16)


            with torch.no_grad():
                # Remove temperature if do_sample is False
                output = model.generate(**final_inputs, max_new_tokens=20, do_sample=False) 
            
            decoded_full = processor.decode(output[0], skip_special_tokens=True).strip()
            
            raw_answer = decoded_full.lower().split("assistant")[-1].strip() if "assistant" in decoded_full.lower() else decoded_full.lower()
            parsed_answer = raw_answer.replace('.', '').replace(',', '').strip()

            if parsed_answer == "true":
                llm_interaction_result = True; llm_answer_display = "TRUE"
            elif parsed_answer == "false":
                llm_interaction_result = False; llm_answer_display = "FALSE"
            else:
                if "true" in parsed_answer: llm_interaction_result = True; llm_answer_display = "TRUE (inf)"
                elif "false" in parsed_answer: llm_interaction_result = False; llm_answer_display = "FALSE (inf)"
                else: llm_interaction_result = False; llm_answer_display = "UNKNOWN"
                print(f"Warning: Frame {frame_idx_str} - LLaVA unclear: '{raw_answer}'. Full: '{decoded_full}' -> Parsed as {llm_answer_display}")
        
        llm_interactions[frame_idx_str] = llm_interaction_result
        
        cv2.putText(annotated_frame_for_video, f"LLM Interact: {llm_answer_display}", (W - 350, H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame_for_video, f"Frame: {frame_idx_str} (Pat: {selected_patient_id}, Hand: {hand_to_track.upper()})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200),2)

        out_video_writer.write(annotated_frame_for_video)

    # --- Cleanup and Save ---
    cap.release()
    out_video_writer.release()
    print(f"✅ Annotated video saved to: {output_video_path}")

    try:
        with open(output_roi_details_json, "w") as f: json.dump(llm_roi_details, f, indent=2)
        print(f"✅ LLM ROI details saved to: {output_roi_details_json}")
    except IOError as e: print(f"Error saving ROI details JSON: {e}")

    try:
        with open(output_interactions_json, "w") as f: json.dump(llm_interactions, f, indent=2)
        print(f"✅ LLaVA interaction log saved to: {output_interactions_json}")
    except IOError as e: print(f"Error saving LLaVA interactions JSON: {e}")

    print("--- LLM Interaction Analysis (ROI around Wrist) Complete ---")
    return True


if __name__ == "__main__":
    # These would be set by your notebook's previous cells or a master script
    _SELECTED_PATIENT_ID = 0  
    _HAND_TO_TRACK = "R"      # "L" or "R"
    _VIDEO_PATH = "" # !!! REPLACE THIS !!!
    
    _MAPPED_HAND_DATA_JSON_PATH = f"patient_{_SELECTED_PATIENT_ID}_hand_{_HAND_TO_TRACK.lower()}_mapped_data.json" 
    
    _OUTPUT_LLM_VIDEO = f"final_llm_wrist_roi_pat{_SELECTED_PATIENT_ID}_{_HAND_TO_TRACK.lower()}_video.mp4"
    _OUTPUT_LLM_ROI_JSON = f"final_llm_wrist_roi_pat{_SELECTED_PATIENT_ID}_{_HAND_TO_TRACK.lower()}_roi_details.json"
    _OUTPUT_LLM_INTERACTIONS_JSON = f"final_llm_wrist_roi_pat{_SELECTED_PATIENT_ID}_{_HAND_TO_TRACK.lower()}_interactions.json"

    # Create dummy mapped_hand_data_enhanced.json if it doesn't exist for testing
    if not os.path.exists(_MAPPED_HAND_DATA_JSON_PATH) and _VIDEO_PATH != "":
        print(f"Warning: Mapped hand data '{_MAPPED_HAND_DATA_JSON_PATH}' not found. Creating a dummy file for testing.")
        dummy_mapped_data = {}
        for i in range(50): 
            dummy_mapped_data[str(i)] = {
                "selected_patient_id": _SELECTED_PATIENT_ID,
                "patient_bbox": [100,100,400,400],
                "tracked_hand_label": _HAND_TO_TRACK,
                "target_wrist_coords": [250.0,250.0] if _HAND_TO_TRACK == "R" else [150.0,250.0],
                "target_wrist_confidence": 0.95,
                "assigned_hand_box": None, 
                "assigned_hand_center": None,
                "assigned_hand_smoothed_center": None,
            }
        with open(_MAPPED_HAND_DATA_JSON_PATH, "w") as f: json.dump(dummy_mapped_data, f, indent=2)


    if _VIDEO_PATH == "path/to/your/input_video.mp4":
        print("\nERROR: Please update '_VIDEO_PATH' in the __main__ block.")
    else:
        llm_interaction_analyzer(
            selected_patient_id=_SELECTED_PATIENT_ID,
            hand_to_track=_HAND_TO_TRACK,
            video_path=_VIDEO_PATH,
            mapped_hand_data_path=_MAPPED_HAND_DATA_JSON_PATH,
            output_video_path=_OUTPUT_LLM_VIDEO,
            output_roi_details_json=_OUTPUT_LLM_ROI_JSON,
            output_interactions_json=_OUTPUT_LLM_INTERACTIONS_JSON,
            wrist_roi_padding=DEFAULT_WRIST_ROI_PADDING, 
            min_wrist_confidence_for_roi=DEFAULT_MIN_WRIST_CONFIDENCE_FOR_ROI,
            save_debug_frames=True, 
            debug_frame_interval=25 
        )