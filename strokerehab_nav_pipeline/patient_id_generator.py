import json
import numpy as np
from typing import List, Dict, Tuple, Any


def iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection over Union between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0

def pid_generator(JSON_INPUT_FILE: str = "output_video_data.json", JSON_OUTPUT_FILE: str = "framewise_identity_patient_ids_enhanced.json", IOU_THRESHOLD: float = 0.5, MAX_AGE_THRESHOLD: int = 15):
    try:
        with open(JSON_INPUT_FILE, "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file '{JSON_INPUT_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{JSON_INPUT_FILE}'.")
        return

    # Restructure data: key by re-indexed frame, value is list of patient boxes
    # Handles original frame indices which might be strings or non-sequential if FRAME_STRIDE > 1
    # And extracts only the "box" part.
    processed_frame_detections: Dict[int, List[Dict[str, Any]]] = {}
    sorted_original_frame_indices = sorted(raw_data.keys(), key=lambda x: int(x))

    for re_indexed_frame, original_frame_idx_str in enumerate(sorted_original_frame_indices):
        frame_content = raw_data[original_frame_idx_str]
        # Assuming 'patient_boxes' is a list of dicts: [{"box": [coords], "score": val}, ...]
        patient_data_list = frame_content.get("patient_boxes", [])
        # We only need the box for IoU, but might keep score for future use if needed
        processed_frame_detections[re_indexed_frame] = [
            {"box": item["box"], "score": item.get("score", 0.0)}
            for item in patient_data_list if "box" in item
        ]

    # Tracks: dictionary where key is track_id, value is a dict of its properties
    # e.g., {track_id: {"box": [coords], "age": 0, "last_update_frame": frame_idx, "id_str": "patient_X"}}
    active_tracks: Dict[int, Dict[str, Any]] = {}
    next_available_track_id: int = 0

    # Stores the state of active_tracks for each frame
    framewise_identity_log: Dict[int, Dict[int, Dict[str, Any]]] = {}

    if not processed_frame_detections:
        print("No patient boxes found in the input data.")
        # Save an empty log if desired
        with open(JSON_OUTPUT_FILE, "w") as f:
            json.dump({"log": framewise_identity_log, "message": "No patient data found"}, f, indent=4)
        return

    print(f"Starting patient tracking. IoU Threshold: {IOU_THRESHOLD}, Max Age: {MAX_AGE_THRESHOLD}")

    for frame_idx_reindexed in range(len(processed_frame_detections)):
        current_frame_patient_detections = processed_frame_detections.get(frame_idx_reindexed, [])

        # --- 1. Age existing tracks and predict (prediction not implemented here, just aging) ---
        tracks_to_delete = []
        for track_id in active_tracks:
            active_tracks[track_id]["age"] += 1
            if active_tracks[track_id]["age"] > MAX_AGE_THRESHOLD:
                tracks_to_delete.append(track_id)

        # Prune old tracks
        for track_id in tracks_to_delete:
            # print(f"Frame {frame_idx_reindexed}: Pruning track {active_tracks[track_id]['id_str']} (age: {active_tracks[track_id]['age']})")
            del active_tracks[track_id]

        # --- 2. Match current detections to existing tracks ---
        unmatched_detections_indices = list(range(len(current_frame_patient_detections)))
        matched_track_ids_this_frame = set()

        # Try to match each active track with a detection
        # (Could also iterate detections first, then tracks - order can matter in greedy assignment)
        # For simplicity, let's iterate detections and try to match them to existing tracks.
        
        current_detections_with_matches: List[Tuple[int, int, float]] = [] # (detection_idx, track_id, iou_score)

        for det_idx, detection_dict in enumerate(current_frame_patient_detections):
            detected_box = detection_dict["box"]
            best_match_iou = 0.0
            best_match_track_id = -1

            for track_id, track_data in active_tracks.items():
                if track_id in matched_track_ids_this_frame: # Track already matched this frame
                    continue
                
                current_iou = iou(detected_box, track_data["box"])
                if current_iou > best_match_iou:
                    best_match_iou = current_iou
                    best_match_track_id = track_id
            
            if best_match_iou >= IOU_THRESHOLD:
                current_detections_with_matches.append((det_idx, best_match_track_id, best_match_iou))

        # Resolve multiple detections matching the same track (take best IoU for that track)
        # Or one detection matching multiple tracks (already handled by finding *best* match for detection)
        # For now, a simple greedy assignment: if a track is matched multiple times, the detection with highest IoU wins.
        # More robustly, one might use Hungarian algorithm here for optimal assignment.
        
        # To ensure a track is only updated once by the best detection for it:
        track_best_detection_match: Dict[int, Tuple[int, float]] = {} # track_id -> (detection_idx, iou_score)

        for det_idx, track_id, iou_score in current_detections_with_matches:
            if track_id not in track_best_detection_match or iou_score > track_best_detection_match[track_id][1]:
                track_best_detection_match[track_id] = (det_idx, iou_score)
        
        temp_unmatched_detections_indices = list(range(len(current_frame_patient_detections)))

        for track_id, (det_idx, iou_score) in track_best_detection_match.items():
            detection_dict = current_frame_patient_detections[det_idx]
            active_tracks[track_id]["box"] = detection_dict["box"]
            active_tracks[track_id]["age"] = 0 # Reset age
            active_tracks[track_id]["last_update_frame"] = frame_idx_reindexed
            active_tracks[track_id]["score"] = detection_dict.get("score") # Update score if available
            matched_track_ids_this_frame.add(track_id)
            if det_idx in temp_unmatched_detections_indices:
                temp_unmatched_detections_indices.remove(det_idx)
        
        unmatched_detections_indices = temp_unmatched_detections_indices


        # --- 3. Create new tracks for unmatched detections ---
        for det_idx in unmatched_detections_indices:
            detection_dict = current_frame_patient_detections[det_idx]
            new_id_str = f"patient_{next_available_track_id}"
            # print(f"Frame {frame_idx_reindexed}: New track {new_id_str} for detection {det_idx}")
            active_tracks[next_available_track_id] = {
                "box": detection_dict["box"],
                "age": 0,
                "last_update_frame": frame_idx_reindexed,
                "id_str": new_id_str,
                "score": detection_dict.get("score")
            }
            next_available_track_id += 1

        # --- 4. Log the state of active_tracks for this frame ---
        # Store a deep copy of the track data for this frame
        # Original frame index from input JSON is more user-friendly for keys in log.
        original_frame_key = sorted_original_frame_indices[frame_idx_reindexed]
        framewise_identity_log[original_frame_key] = {
            tid: data.copy() for tid, data in active_tracks.items()
        }

    # --- Save the results ---
    try:
        with open(JSON_OUTPUT_FILE, "w") as f:
            json.dump(framewise_identity_log, f, indent=4)
        print(f"✅ Done! Enhanced tracking data saved to: {JSON_OUTPUT_FILE}")
    except IOError as e:
        print(f"Error saving enhanced JSON data: {e}")

    # You can return this if you want to use it directly
    return framewise_identity_log

if __name__ == "__main__":
    tracked_data = pid_generator()