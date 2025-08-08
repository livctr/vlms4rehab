import json
import numpy as np
from collections import defaultdict, deque
import math
import os
import sys
from typing import Optional, Tuple, Dict, Any, Deque, List
from tqdm import tqdm
# --- Attempt to import OpenCV ---
cv2 = None
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV (cv2) library not found. Visualization will be disabled.")
    CV2_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error importing OpenCV ({e}). Visualization will be disabled.")
    CV2_AVAILABLE = False

# --- Constants and Configuration ---
# These can be adjusted as needed
MAX_WRIST_HAND_DISTANCE = 150  # Max distance (pixels) between target wrist and hand center. Adjusted for potentially tighter coupling.
DISAMBIGUATION_FACTOR = 0.7 # Hand must be this factor closer to target wrist than other wrist.
SMOOTHING_WINDOW_SIZE = 7    # Frames for hand centroid smoothing.
MIN_WRIST_CONFIDENCE = 0.3  # Minimum confidence for a wrist keypoint from pose data to be considered reliable.

# --- Helper Functions ---
def calculate_center(box: List[float]) -> Optional[np.ndarray]:
    """Calculates the center of a bounding box. Returns None on error or invalid box."""
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(c, (int, float)) and math.isfinite(c) for c in box)):
        return None
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2: # Check for valid box dimensions
        # print(f"Warning: Invalid box dimensions for center calculation: {box}")
        return None
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def distance(point1: Optional[np.ndarray], point2: Optional[np.ndarray]) -> float:
    """Calculates Euclidean distance. Returns float('inf') if any point is None or invalid."""
    if point1 is None or point2 is None:
        return float('inf')
    try:
        p1_arr = np.array(point1, dtype=float)
        p2_arr = np.array(point2, dtype=float)
        if p1_arr.shape == (2,) and p2_arr.shape == (2,) and np.all(np.isfinite(p1_arr)) and np.all(np.isfinite(p2_arr)):
            return np.linalg.norm(p1_arr - p2_arr)
        return float('inf')
    except Exception:
        return float('inf')

def is_inside(point: Optional[np.ndarray], box: Optional[List[float]]) -> bool:
    """Checks if a point (x, y) is inside a box [x1, y1, x2, y2]."""
    if point is None or box is None: return False
    if not (len(point) == 2 and len(box) == 4 and all(isinstance(c, (int, float)) for c in box)): return False
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# --- Main Enhanced Function ---
def map_hand_to_patient_enhanced(
    selected_patient_id: int,
    hand_to_track: str, # "L" or "R"
    video_path: Optional[str],
    pose_data_path: str = "keypoints_enhanced_output.json", # From enhanced_extract_pose_keypoints
    hand_detections_path: str = "output_video_data.json", # From initial object detection
    output_json_path: str = "mapped_hand_data_enhanced.json",
    output_video_path: Optional[str] = "tracked_hand_video_enhanced.mp4",
    max_wrist_hand_dist: float = MAX_WRIST_HAND_DISTANCE,
    disamb_factor: float = DISAMBIGUATION_FACTOR,
    smoothing_window: int = SMOOTHING_WINDOW_SIZE,
    min_wrist_conf: float = MIN_WRIST_CONFIDENCE
):
    """
    Maps detected hand bounding boxes to a specific patient's tracked wrist (L or R)
    using pose estimation data and hand detection data.

    Args:
        selected_patient_id: The integer ID of the patient to focus on.
        hand_to_track: Which hand to track for the selected patient ("L" or "R").
        video_path: Path to the original video for visualization (optional).
        pose_data_path: Path to JSON from enhanced pose extraction (contains patient bbox & wrist kps).
        hand_detections_path: Path to JSON from general object detection (contains hand_boxes).
        output_json_path: Path to save the resulting hand tracking data.
        output_video_path: Path to save the annotated video (if visualization is enabled).
        max_wrist_hand_dist: Maximum allowed distance between target wrist and hand center.
        disamb_factor: Hand must be this factor closer to target wrist than the other wrist.
        smoothing_window: Window size for hand centroid smoothing.
        min_wrist_conf: Minimum confidence for wrist keypoints from pose data to be used.
    """

    print(f"--- Starting Hand Mapping for Patient ID: {selected_patient_id}, Hand: {hand_to_track.upper()} ---")
    if hand_to_track.upper() not in ["L", "R"]:
        print(f"Error: 'hand_to_track' must be 'L' or 'R'. Received: {hand_to_track}")
        sys.exit(1)

    target_label = hand_to_track.upper()

    required_files = [pose_data_path, hand_detections_path]
    missing_required = [f for f in required_files if not os.path.exists(f)]
    if missing_required:
        print(f"Error: Missing required input file(s): {', '.join(missing_required)}")
        sys.exit(1)

    print("Loading data...")
    try:
        with open(pose_data_path, "r") as f: pose_data_all_frames = json.load(f)
        with open(hand_detections_path, "r") as f: hand_detect_data_all_frames = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    print("Data loading complete.")

    visualize = CV2_AVAILABLE and video_path is not None
    cap, out_vid, W, H = None, None, 0, 0
    if visualize:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise IOError(f"Cannot open video file: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS); fps = 30 if fps <= 0 else fps
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if W <= 0 or H <= 0: raise ValueError("Invalid video dimensions from video file.")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
            if not out_vid.isOpened():
                print(f"Warning: Could not open video writer with MP4V codec for {output_video_path}. Trying XVID/AVI.")
                output_video_path = output_video_path.replace(".mp4", ".avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
                if not out_vid.isOpened(): raise IOError("Could not open video writer with any codec.")
            print(f"Video setup successful for visualization. Output: {output_video_path}")
        except Exception as e:
            print(f"Warning: Video setup failed ({e}). Disabling visualization.")
            visualize = False
            if cap and cap.isOpened(): cap.release()
            if out_vid and out_vid.isOpened(): out_vid.release() # Ensure it's released if opened
            cap, out_vid = None, None


    framewise_tracked_hand_output: Dict[str, Dict[str, Any]] = defaultdict(dict)
    center_history: Deque[np.ndarray] = deque(maxlen=smoothing_window)
    processed_frames_count = 0

    # Iterate through frames where pose data for *any* patient is available
    # We will then filter for the selected_patient_id
    sorted_frame_keys_from_pose = sorted(pose_data_all_frames.keys(), key=int)

    print(f"Processing {len(sorted_frame_keys_from_pose)} frames based on pose data...")
    for frame_id_str in tqdm(sorted_frame_keys_from_pose, desc="Mapping hands to patient"):
        frame_num_int = int(frame_id_str)
        
        current_pose_info = pose_data_all_frames.get(frame_id_str)
        if not current_pose_info or current_pose_info.get("patient_id_track") != selected_patient_id:
            # This frame in pose_data is not for our selected patient, or data is missing
            if visualize and cap and out_vid: # Write original frame if visualizing
                ret, frame = cap.read()
                if ret: out_vid.write(frame)
                else: break # End of video
            elif cap: # If not visualizing video but need to advance frame
                 cap.grab()
            frame_idx_from_pose_json = current_pose_info.get("patient_id_track", "Unknown") if current_pose_info else "Unknown"
            # print(f"Skipping frame {frame_id_str}: Pose data not for patient {selected_patient_id} (found for {frame_idx_from_pose_json}) or data missing.")
            continue

        processed_frames_count += 1
        frame = None
        if visualize and cap:
            # It's important to ensure cap.read() is synchronized with frame_id_str.
            # If pose_data has gaps, we need to skip frames in video too.
            # A robust way is to set frame position, but can be slow.
            # Assuming pose_data frames are sequential from video for this example.
            # For production, you might build a frame map or always use cap.set()
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != frame_num_int:
                 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_int)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read video frame {frame_id_str} for visualization. Stopping vis.")
                visualize = False # Stop visualization

        patient_box_coords = current_pose_info.get("patient_bbox_in_frame")
        
        target_wrist_pt: Optional[np.ndarray] = None
        other_wrist_pt: Optional[np.ndarray] = None
        target_wrist_conf: Optional[float] = None

        if target_label == 'L':
            if current_pose_info.get("left_wrist_smooth") and current_pose_info.get("left_wrist_conf", 0) >= min_wrist_conf:
                target_wrist_pt = np.array(current_pose_info["left_wrist_smooth"])
                target_wrist_conf = current_pose_info["left_wrist_conf"]
            if current_pose_info.get("right_wrist_smooth") and current_pose_info.get("right_wrist_conf", 0) >= min_wrist_conf:
                other_wrist_pt = np.array(current_pose_info["right_wrist_smooth"])
        else: # target_label == 'R'
            if current_pose_info.get("right_wrist_smooth") and current_pose_info.get("right_wrist_conf", 0) >= min_wrist_conf:
                target_wrist_pt = np.array(current_pose_info["right_wrist_smooth"])
                target_wrist_conf = current_pose_info["right_wrist_conf"]
            if current_pose_info.get("left_wrist_smooth") and current_pose_info.get("left_wrist_conf", 0) >= min_wrist_conf:
                other_wrist_pt = np.array(current_pose_info["left_wrist_smooth"])

        # Get hand detections for this frame
        frame_hand_detections_data = hand_detect_data_all_frames.get(frame_id_str, {})
        # Assuming hand_boxes is a list of boxes: [[x1,y1,x2,y2], ...] or list of dicts [{"box": [...]},...]
        raw_hand_boxes = frame_hand_detections_data.get("hand_boxes", [])
        current_frame_hand_boxes = []
        if raw_hand_boxes and isinstance(raw_hand_boxes[0], dict): # It's list of dicts
            current_frame_hand_boxes = [item["box"] for item in raw_hand_boxes if "box" in item]
        elif raw_hand_boxes and isinstance(raw_hand_boxes[0], list): # It's list of lists
            current_frame_hand_boxes = raw_hand_boxes


        assigned_hand_box: Optional[List[float]] = None
        current_center: Optional[np.ndarray] = None
        min_dist_to_target_wrist = float('inf')
        best_candidate_hand_box = None
        best_candidate_center = None

        if target_wrist_pt is not None and current_frame_hand_boxes:
            for hand_box in current_frame_hand_boxes:
                hand_center = calculate_center(hand_box)
                if hand_center is None: continue

                if patient_box_coords and not is_inside(hand_center, patient_box_coords):
                    continue

                dist_target = distance(hand_center, target_wrist_pt)
                if dist_target < min_dist_to_target_wrist:
                    min_dist_to_target_wrist = dist_target
                    best_candidate_hand_box = hand_box
                    best_candidate_center = hand_center
            
            if best_candidate_box is not None and min_dist_to_target_wrist <= max_wrist_hand_dist:
                assign_this_hand = True
                if other_wrist_pt is not None: # Disambiguation
                    dist_other = distance(best_candidate_center, other_wrist_pt)
                    if min_dist_to_target_wrist >= dist_other * disamb_factor: # Check if it's too close to the wrong wrist
                        assign_this_hand = False
                
                if assign_this_hand:
                    assigned_hand_box = best_candidate_box
                    current_center = best_candidate_center

        # Update smoothing history and get smoothed center
        smoothed_center: Optional[np.ndarray] = None
        if current_center is not None:
            center_history.append(current_center)
        # Only smooth if there's history; don't clear on one miss for less jumpy behavior.
        # If no current_center, the history will naturally phase out old values.
        if len(center_history) > 0:
            try:
                smoothed_center = np.mean(center_history, axis=0)
            except Exception: # Should not happen with np.mean on deque of np.arrays
                smoothed_center = center_history[-1] if center_history else None
        
        # --- Store Results for this frame ---
        output_entry = {
            "selected_patient_id": selected_patient_id,
            "patient_bbox": patient_box_coords,
            "tracked_hand_label": target_label,
            "target_wrist_coords": target_wrist_pt.tolist() if target_wrist_pt is not None else None,
            "target_wrist_confidence": target_wrist_conf,
            "other_wrist_coords": other_wrist_pt.tolist() if other_wrist_pt is not None else None,
            "assigned_hand_box": [float(c) for c in assigned_hand_box] if assigned_hand_box else None,
            "assigned_hand_center": current_center.tolist() if current_center is not None else None,
            "assigned_hand_smoothed_center": smoothed_center.tolist() if smoothed_center is not None else None
        }
        framewise_tracked_hand_output[frame_id_str] = output_entry

        # --- Visualization ---
        if visualize and frame is not None and out_vid is not None:
            try:
                if patient_box_coords:
                    px1, py1, px2, py2 = map(int, patient_box_coords)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 1) # Green for patient
                    cv2.putText(frame, f"Patient {selected_patient_id}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                if target_wrist_pt is not None:
                    cv2.circle(frame, tuple(target_wrist_pt.astype(int)), 7, (255, 255, 0), -1) # Cyan dot for target wrist
                    cv2.putText(frame, f"{target_label}w", tuple((target_wrist_pt + np.array([8,0])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                if other_wrist_pt is not None:
                    cv2.circle(frame, tuple(other_wrist_pt.astype(int)), 4, (255, 0, 255), -1) # Magenta for other wrist

                if assigned_hand_box:
                    hx1, hy1, hx2, hy2 = map(int, assigned_hand_box)
                    hand_color = (255,0,0) if target_label == 'L' else (0,0,255) # Blue for Left, Red for Right
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), hand_color, 2)
                    cv2.putText(frame, f"Tracked {target_label}-Hand", (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)
                    if smoothed_center is not None:
                        cv2.circle(frame, tuple(smoothed_center.astype(int)), 5, (0, 255, 0), -1) # Green for smoothed center
                        if current_center is not None: # Draw raw center as smaller dot
                             cv2.circle(frame, tuple(current_center.astype(int)), 3, (0,128,0), -1)


                cv2.putText(frame, f"F: {frame_id_str}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                out_vid.write(frame)
            except Exception as e:
                print(f"Error during visualization for frame {frame_id_str}: {e}")
        elif visualize and not ret and cap: # End of video frames but loop continues on JSON keys
             print(f"Note: Reached end of video frames for visualization at frame {frame_id_str}.")
             visualize = False # Stop trying to visualize


    print(f"Finished processing. Total frames processed for patient {selected_patient_id}: {processed_frames_count}.")

    if cap: cap.release()
    if out_vid: out_vid.release()
    if CV2_AVAILABLE and visualize and output_video_path: # Check if output_video_path was set
        print(f"Annotated video saved to {output_video_path}")
    elif CV2_AVAILABLE and video_path and not output_video_path:
        print(f"Visualization was attempted but video writer failed to open.")


    print(f"Saving tracked hand data to {output_json_path}...")
    if not framewise_tracked_hand_output:
        print("Warning: No tracked hand data was generated.")
    else:
        try:
            # Sort output by frame number for consistency
            final_sorted_output = dict(sorted(framewise_tracked_hand_output.items(), key=lambda item: int(item[0])))
            with open(output_json_path, "w") as f:
                json.dump(final_sorted_output, f, indent=2)
            print(f"Data saved successfully to {output_json_path}.")
        except Exception as e:
            print(f"Error saving JSON data: {e}")
            return False
    return True