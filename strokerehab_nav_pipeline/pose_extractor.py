import torch
from ultralytics import YOLO
import cv2
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, Any, Deque, List

# --- Constants ---
LEFT_WRIST_KP_IDX = 9
RIGHT_WRIST_KP_IDX = 10
MIN_KP_CONFIDENCE = 0.3 # Minimum confidence for a keypoint to be considered reliable

# --- Helper Function for Wrist Processing ---
def _process_single_wrist(
    wrist_kp_data: np.ndarray, # Shape (3,) [x, y, confidence]
    history_deque: Deque[np.ndarray],
    last_known_pos: Optional[np.ndarray],
    jump_threshold: float,
    min_confidence: float,
    patient_bbox: Tuple[int, int, int, int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Processes a single wrist keypoint: checks confidence, location, handles jumps, and smooths.

    Args:
        wrist_kp_data: Numpy array [x, y, confidence] for the current wrist detection.
        history_deque: Deque storing recent valid positions for smoothing.
        last_known_pos: Last known reliable position of this wrist.
        jump_threshold: Max distance for a jump to be corrected.
        min_confidence: Minimum confidence score for the keypoint to be considered valid.
        patient_bbox: The bounding box (x1, y1, x2, y2) of the target patient.

    Returns:
        A tuple (smoothed_wrist_pos, current_valid_wrist_pos).
        smoothed_wrist_pos is [x,y] or None.
        current_valid_wrist_pos is the [x,y] position if valid this frame, else None.
    """
    x1_pat, y1_pat, x2_pat, y2_pat = patient_bbox
    wrist_pos = wrist_kp_data[:2]
    wrist_conf = wrist_kp_data[2]

    current_valid_pos = None

    if wrist_conf >= min_confidence:
        # Check if the wrist is inside the patient's bounding box
        if x1_pat <= wrist_pos[0] <= x2_pat and y1_pat <= wrist_pos[1] <= y2_pat:
            current_valid_pos = wrist_pos.copy() # Store current valid raw position

            # Jump detection and correction
            # Use last_known_pos for jump comparison, not mean of history, to avoid drift from bad initial points
            comparison_pos = last_known_pos if last_known_pos is not None else current_valid_pos
            
            if last_known_pos is not None and np.linalg.norm(current_valid_pos - comparison_pos) > jump_threshold:
                # If jumped, use the last known position (or mean of history for smoother recovery)
                if history_deque: # Use smoothed history if available
                    current_valid_pos = np.mean(history_deque, axis=0)
                else: # Fallback to last raw known position
                    current_valid_pos = last_known_pos
            
            history_deque.append(current_valid_pos) # Add to history for smoothing
            # last_known_pos will be updated with this current_valid_pos outside this function
    
    # If current_valid_pos is None (due to low confidence or outside bbox),
    # try to use history or last_known_pos for drawing, but don't update history with it.
    smoothed_pos = None
    if history_deque:
        smoothed_pos = np.mean(history_deque, axis=0)
    elif last_known_pos is not None: # If no history but had a previous good point
        smoothed_pos = last_known_pos
        
    return smoothed_pos, current_valid_pos


def extract_pose_keypoints(
    tracker_hand: str = "L", # "L", "R", or "BOTH"
    pid_json_path: str = "framewise_identity_patient_ids_enhanced.json",
    output_json_path: str = "pose_kpts_enhanced.json",
    output_video_path: str = "pose_output_enhanced.mp4",
    jump_threshold: float = 50.0, # Max distance for a jump (pixels)
    smooth_window: int = 10,
    patient_id_track: int = 0, # The integer ID of the patient to track
    video_path: str = "video_test_patient_.mp4",
    min_keypoint_confidence: float = MIN_KP_CONFIDENCE,
    yolo_pose_model_path: str = "yolov8l-pose.pt" # Example: yolov8s-pose.pt, yolov8m-pose.pt etc.
):
    """
    Extracts pose keypoints for a specified patient, focusing on wrist tracking.

    Args:
        tracker_hand: Which hand(s) to track: "L", "R", or "BOTH".
        pid_json_path: Path to the JSON file containing patient tracking data (from pid_generator).
        output_json_path: Path to save the extracted pose keypoints.
        output_video_path: Path to save the annotated video.
        jump_threshold: Max distance for a wrist movement to be considered a jump and corrected.
                        A very large value (e.g., 10000) effectively disables jump correction.
        smooth_window: Number of frames for the moving average smoothing of wrist positions.
        patient_id_track: The integer ID of the patient whose pose is to be tracked.
        video_path: Path to the input video file.
        min_keypoint_confidence: Minimum confidence for detected keypoints to be considered.
        yolo_pose_model_path: Path to the YOLOv8 Pose model file (e.g., yolov8s-pose.pt).
    """
    print(f"Loading YOLO Pose model from: {yolo_pose_model_path}")
    model = YOLO(yolo_pose_model_path)
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    try:
        print(f"Loading patient ID data from: {pid_json_path}")
        with open(pid_json_path) as f:
            pid_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Patient ID JSON file not found at {pid_json_path}")
        cap.release()
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {pid_json_path}")
        cap.release()
        return False

    frame_idx = 0
    all_frames_pose_data: Dict[str, Dict[str, Any]] = {} # Stores pose data for JSON output

    # History for smoothing and last known positions for jump correction
    # Key: "L" or "R" for the specific patient_id_track
    wrist_history = {
        "L": deque(maxlen=smooth_window),
        "R": deque(maxlen=smooth_window)
    }
    # Stores the last *raw valid* position [x,y] before smoothing, for jump reference.
    wrist_last_valid_raw = {"L": None, "R": None}


    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_vid = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {total_frames} frames, FPS: {fps}, Resolution: {W}x{H}")
    print(f"Tracking patient ID: {patient_id_track}, Hand(s): {tracker_hand.upper()}")
    if jump_threshold > W : # Heuristic for very large jump threshold
         print(f"Warning: jump_threshold ({jump_threshold}) is very large, jump correction may be ineffective.")


    for _ in tqdm(range(total_frames), desc="Processing frames for pose"):
        ret, frame = cap.read()
        if not ret:
            break

        # Get patient's bounding box for the current frame
        patient_bbox_data_for_frame = pid_data.get(str(frame_idx)) # Original frame index is string
        patient_specific_data = None
        if patient_bbox_data_for_frame:
            # patient_id_track is an int, keys in pid_data[frame] are str(int_track_id)
            patient_specific_data = patient_bbox_data_for_frame.get(str(patient_id_track))

        current_frame_output_data: Dict[str, Any] = {
            "patient_id_track": patient_id_track,
            "patient_bbox_in_frame": None,
            "yolo_person_index": None,
            "raw_keypoints": None,
            "left_wrist_smooth": None, "left_wrist_conf": None,
            "right_wrist_smooth": None, "right_wrist_conf": None,
        }

        if patient_specific_data and "box" in patient_specific_data:
            pat_box_coords = patient_specific_data["box"]
            x1, y1, x2, y2 = map(int, pat_box_coords)
            current_frame_output_data["patient_bbox_in_frame"] = [x1, y1, x2, y2]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1) # Draw patient bbox

            # Run YOLO pose detection on the current frame
            # Consider cropping frame to patient_bbox + padding before predict for performance,
            # but this might affect detection quality if context is lost.
            results = model.predict(frame, verbose=False, conf=0.3) # Overall detection confidence
            
            processed_this_person_for_L = False
            processed_this_person_for_R = False

            # Iterate through all persons detected in the frame
            for p_idx, person_kps_data in enumerate(results[0].keypoints.data):
                if person_kps_data is None:
                    continue

                kpts_np = person_kps_data.cpu().numpy() # Shape: (num_kps, 3) [x, y, conf]
                
                # Check if this person is likely our target patient based on keypoint proximity to bbox
                # (A simple check: if a major keypoint like nose or a shoulder is within the box)
                # For now, we simply check if their *wrists* fall inside the box.

                smoothed_lw, current_valid_lw_raw = None, None
                smoothed_rw, current_valid_rw_raw = None, None

                # Process Left Wrist
                if tracker_hand.upper() in ["L", "BOTH"] and not processed_this_person_for_L:
                    lw_data = kpts_np[LEFT_WRIST_KP_IDX] # [x, y, conf]
                    smoothed_lw, current_valid_lw_raw = _process_single_wrist(
                        lw_data, wrist_history["L"], wrist_last_valid_raw["L"],
                        jump_threshold, min_keypoint_confidence, (x1,y1,x2,y2)
                    )
                    if current_valid_lw_raw is not None: # Successfully tracked this person's L wrist
                        wrist_last_valid_raw["L"] = current_valid_lw_raw
                        processed_this_person_for_L = True # Mark this yolo person as processed for L wrist
                        current_frame_output_data["yolo_person_index"] = p_idx # Associate this person
                        current_frame_output_data["raw_keypoints"] = kpts_np.tolist()
                        current_frame_output_data["left_wrist_smooth"] = smoothed_lw.tolist() if smoothed_lw is not None else None
                        current_frame_output_data["left_wrist_conf"] = float(lw_data[2]) if current_valid_lw_raw is not None else None


                # Process Right Wrist
                if tracker_hand.upper() in ["R", "BOTH"] and not processed_this_person_for_R:
                    rw_data = kpts_np[RIGHT_WRIST_KP_IDX] # [x, y, conf]
                    smoothed_rw, current_valid_rw_raw = _process_single_wrist(
                        rw_data, wrist_history["R"], wrist_last_valid_raw["R"],
                        jump_threshold, min_keypoint_confidence, (x1,y1,x2,y2)
                    )
                    if current_valid_rw_raw is not None: # Successfully tracked this person's R wrist
                        wrist_last_valid_raw["R"] = current_valid_rw_raw
                        processed_this_person_for_R = True # Mark this yolo person as processed for R wrist
                        # If L wrist didn't set this, or if it's the same person
                        if current_frame_output_data["yolo_person_index"] is None or current_frame_output_data["yolo_person_index"] == p_idx:
                            current_frame_output_data["yolo_person_index"] = p_idx
                            current_frame_output_data["raw_keypoints"] = kpts_np.tolist()
                        current_frame_output_data["right_wrist_smooth"] = smoothed_rw.tolist() if smoothed_rw is not None else None
                        current_frame_output_data["right_wrist_conf"] = float(rw_data[2]) if current_valid_rw_raw is not None else None
                
                # If this person was matched for both (if BOTH selected), we can break early if desired,
                # or continue if other people might also have wrists in the box (less likely for tight boxes).
                # For now, we assume one main person of interest.
                if processed_this_person_for_L and processed_this_person_for_R and tracker_hand.upper() == "BOTH":
                    break 
                if processed_this_person_for_L and tracker_hand.upper() == "L":
                    break
                if processed_this_person_for_R and tracker_hand.upper() == "R":
                    break
            
            # --- Drawing ---
            # Draw Left Wrist if tracked and smoothed
            final_lw_to_draw = current_frame_output_data["left_wrist_smooth"]
            if final_lw_to_draw:
                cv2.circle(frame, tuple(np.array(final_lw_to_draw).astype(int)), 6, (255, 0, 0), -1) # Blue for Left
                cv2.putText(frame, "L", tuple((np.array(final_lw_to_draw) + 5).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw Right Wrist if tracked and smoothed
            final_rw_to_draw = current_frame_output_data["right_wrist_smooth"]
            if final_rw_to_draw:
                cv2.circle(frame, tuple(np.array(final_rw_to_draw).astype(int)), 6, (0, 0, 255), -1) # Red for Right
                cv2.putText(frame, "R", tuple((np.array(final_rw_to_draw) + 5).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        all_frames_pose_data[str(frame_idx)] = current_frame_output_data
        cv2.putText(frame, f"Frame: {frame_idx}", (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out_vid.write(frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()

    try:
        with open(output_json_path, "w") as f:
            json.dump(all_frames_pose_data, f, indent=2)
        print(f"✅ Pose keypoints saved to {output_json_path}")
    except IOError as e:
        print(f"Error saving pose JSON data: {e}")
        return False

    print(f"✅ Annotated video saved to {output_video_path}")
    return True
