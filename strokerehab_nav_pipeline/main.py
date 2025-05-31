# -*- coding: utf-8 -*-
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm
import json
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import shutil
import yaml
from PIL import Image
import csv
import io
import pdb
from collections import deque
from typing import Dict, Optional, List, Any, Tuple, Union
from Levenshtein import distance as levenshtein_distance


from patient_id_generator import pid_generator
from pose_extractor import extract_pose_keypoints
from hand_id_generator import map_hand_to_patient_enhanced
from llm_caller import llm_interaction_analyzer

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# from new_primitives import calculate_velocity_and_plot # Assuming run_motion_analysis is from here or similar

class LabelUtils:
    PRIMITIVES = ["reach", "reposition", "transport", "stabilize", "idle"]

    @staticmethod
    def get_handedness(path_or_buffer: Union[str, io.StringIO]):
        try:
            if isinstance(path_or_buffer, str) and os.path.exists(path_or_buffer):
                df = pd.read_csv(path_or_buffer)
                path_for_naming_heuristic = path_or_buffer
            elif hasattr(path_or_buffer, 'read'): # Check if it's a file-like object
                df = pd.read_csv(path_or_buffer)
                path_for_naming_heuristic = getattr(path_or_buffer, 'name', '')
            else: # Assume it's a string buffer (though less common for file paths)
                df = pd.read_csv(io.StringIO(str(path_or_buffer)))
                path_for_naming_heuristic = ""
        except Exception as e:
            print(f"Error reading label data for handedness: {e}")
            path_for_naming_heuristic = str(path_or_buffer) if isinstance(path_or_buffer, str) else ""
            if "_r_" in path_for_naming_heuristic or path_for_naming_heuristic.lower().startswith("r_"): return "right"
            if "_l_" in path_for_naming_heuristic or path_for_naming_heuristic.lower().startswith("l_"): return "left"
            return "unknown"

        marker_names = df['MarkerNames'].dropna().astype(str).str.lower()
        num_lefts = marker_names.str.startswith('l_').sum() + marker_names.str.contains('_l_').sum()
        num_rights = marker_names.str.startswith('r_').sum() + marker_names.str.contains('_r_').sum()

        if num_rights > num_lefts: return "right"
        elif num_lefts > num_rights: return "left"
        else:
            if "_r_" in path_for_naming_heuristic or path_for_naming_heuristic.lower().startswith("r_"): return "right"
            if "_l_" in path_for_naming_heuristic or path_for_naming_heuristic.lower().startswith("l_"): return "left"
            print("Warning: Handedness ambiguous from labels and filename. Defaulting to 'right'.")
            return "right"
        return "unknown"


    @staticmethod
    def convert_gt_labels_to_primitive_sequence(label_path: str, handedness: str) -> List[str]:
        if not label_path or not os.path.exists(label_path):
            print("Warning: Ground truth label path not provided or file does not exist for primitive sequence conversion.")
            return []
        try:
            df = pd.read_csv(label_path)
        except Exception as e:
            print(f"Error reading GT label CSV for primitives: {e}")
            return []

        action_seq_primitives = []
        if 'MarkerNames' not in df.columns:
            print(f"Error: 'MarkerNames' column not found in {label_path}")
            return []

        actions = df['MarkerNames'].tolist()
        for action_full_name in actions:
            action_full_name = str(action_full_name).lower()
            primitive_found = None
            is_correct_hand = False
            if handedness == "left" and ("l_" in action_full_name): is_correct_hand = True
            elif handedness == "right" and ("r_" in action_full_name): is_correct_hand = True
            elif handedness not in ["left", "right"]: is_correct_hand = True

            if is_correct_hand:
                if "reach" in action_full_name: primitive_found = "reach"
                elif "reposition" in action_full_name or "retract" in action_full_name: primitive_found = "reposition"
                elif "transport" in action_full_name: primitive_found = "transport"
                elif "stabilize" in action_full_name: primitive_found = "stabilize"
                elif "idle" in action_full_name or "rest" in action_full_name: primitive_found = "idle"

            if primitive_found and (not action_seq_primitives or action_seq_primitives[-1] != primitive_found):
                action_seq_primitives.append(primitive_found)
        return action_seq_primitives

    @staticmethod
    def convert_predicted_primitives_to_sequence(primitive_predictions_list: List[str]) -> List[str]:
        if not primitive_predictions_list: return []
        action_seq = [primitive_predictions_list[0]] if primitive_predictions_list else []
        for prim in primitive_predictions_list[1:]:
            if action_seq[-1] != prim:
                action_seq.append(prim)
        return action_seq

def _get_primitives_score(pred_sequence: List[str], ref_sequence: List[str]) -> Dict:
    pred_sequence = [x.lower() for x in pred_sequence]
    ref_sequence = [x.lower() for x in ref_sequence]
    max_len = max(len(pred_sequence), len(ref_sequence))
    edit_score = 100.
    if max_len > 0:
        temp_pred_str_sequence = "".join([p[0] for p in pred_sequence if p])
        temp_ref_str_sequence = "".join([p[0] for p in ref_sequence if p])
        edit_dist_for_score = levenshtein_distance(temp_pred_str_sequence, temp_ref_str_sequence)
        edit_score = (1 - (edit_dist_for_score / max_len)) * 100.

    action_error_rate = float('inf')
    if len(ref_sequence) > 0:
        edit_dist_for_score = levenshtein_distance("".join([p[0] for p in pred_sequence if p]), "".join([p[0] for p in ref_sequence if p]))
        action_error_rate = edit_dist_for_score / len(ref_sequence)
    elif len(pred_sequence) == 0: # both empty
        action_error_rate = 0.0


    mae_dict = {}
    maes = []
    all_primitives_to_check = sorted(list(set(LabelUtils.PRIMITIVES + pred_sequence + ref_sequence)))
    for primitive in all_primitives_to_check:
        pred_cnt = pred_sequence.count(primitive)
        ref_cnt = ref_sequence.count(primitive)
        mae = abs(pred_cnt - ref_cnt)
        mae_dict[f"{primitive}_mae"] = mae
        if primitive in LabelUtils.PRIMITIVES:
            maes.append(mae)
    avg_mae = sum(maes) / len(LabelUtils.PRIMITIVES) if LabelUtils.PRIMITIVES and maes else (0.0 if not maes else float('nan')) # handle division by zero if no PRIMITIVES
    mae_dict["avg_mae_defined_primitives"] = avg_mae

    return {
        "edit_score_normalized": edit_score,
        "action_error_rate": action_error_rate,
        "mae_per_primitive": mae_dict,
    }



def load_activity_config_from_yaml(yaml_file_path: str) -> Dict[str, str]:
    """Loads activity configuration from a YAML file and returns a dict
       mapping activity name to its target_objects string."""
    activity_to_objects = {}
    try:
        with open(yaml_file_path, 'r') as f:
            activities_config = yaml.safe_load(f)
        if isinstance(activities_config, list):
            for activity_data in activities_config:
                if isinstance(activity_data, dict) and 'name' in activity_data and 'target_objects' in activity_data:
                    activity_to_objects[activity_data['name'].lower()] = activity_data['target_objects']
    except FileNotFoundError:
        print(f"Error: Activity YAML file not found at {yaml_file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading YAML: {e}")
    return activity_to_objects


# MODIFIED run_motion_analysis function
def run_motion_analysis(
    selected_patient_id: int,
    hand_to_track: str,
    pose_data_path: str,
    label_path: Optional[str],
    llm_data_path: Optional[str],
    output_plot_path: str,
    smoothing_window_velocity: int,
    displacement_window_size: int,
    distance_threshold_motion: float,
    min_wrist_confidence: float,
    gt_subsample_factor: int,
    contact_lookahead_window: int = 10) -> Tuple[Optional[pd.DataFrame], Optional[List[Optional[str]]], Optional[Dict], Optional[List[str]], Optional[List[str]]]:
    """
    Performs motion and primitive analysis.
    MODIFIED to return:
    - analysis_df: DataFrame with detailed frame-wise analysis.
    - frame_wise_gt_primitives: List of GT primitives per frame.
    - primitive_scores_dict: Dictionary of calculated primitive scores.
    - predicted_primitives_condensed: Condensed list of predicted primitives.
    - gt_primitives_condensed: Condensed list of GT primitives.
    """
    # ... (Preamble, loading pose data, extracting wrist coords - from previous response) ...
    # ... (Velocity calculation, motion prediction, LLM contact loading - from previous response) ...
    # The key is that `df` is created and populated with `motion_prediction_kinematic`, `contact_llm`,
    # `contact_future`, and `primitive_prediction`.

    # Initialize return values
    analysis_df = None
    frame_wise_gt_primitives_list = []
    primitive_scores_dict = {}
    predicted_primitives_condensed_list = []
    gt_primitives_condensed_list = []

    print(f"\n--- Starting Full Analysis (Motion & Primitives) for patient {selected_patient_id}, hand {hand_to_track} ---")
    # --- Load Pose Keypoint Data ---
    try:
        with open(pose_data_path, "r") as f: pose_data_all_frames = json.load(f)
    except Exception as e:
        print(f"Error loading pose JSON {pose_data_path}: {e}"); return None, None, None, None, None

    # --- Extract Specified Wrist Coordinates & Create initial DataFrame shell ---
    wrist_coords_dict = {} # As before
    # ... (Populate wrist_coords_dict) ...
    # Example population:
    for frame_id_str, frame_data in pose_data_all_frames.items():
        if frame_data.get("patient_id_track") == selected_patient_id:
            key_smooth = f"{'left' if hand_to_track.lower() == 'l' else 'right'}_wrist_smooth"
            key_conf = f"{'left' if hand_to_track.lower() == 'l' else 'right'}_wrist_conf"
            coords = frame_data.get(key_smooth)
            conf = frame_data.get(key_conf)
            if coords and isinstance(coords, list) and len(coords) == 2 and conf is not None and conf >= min_wrist_confidence:
                wrist_coords_dict[int(frame_id_str)] = np.array(coords, dtype=float)

    if not wrist_coords_dict:
        print("No valid wrist coordinates found for the selected patient. Cannot proceed with motion analysis."); return None, None, None, None, None
    
    processed_frame_ids_int = sorted(wrist_coords_dict.keys())
    analysis_df = pd.DataFrame(index=range(min(processed_frame_ids_int), max(processed_frame_ids_int) + 1))
    analysis_df.index.name = 'frame'

    # --- Calculate Velocities, Motion, Contact Future, Predicted Primitives ---
    # Populate analysis_df['raw_velocity'], ['smoothed_velocity'], ['motion_prediction_kinematic'],
    # ['contact_llm'], ['contact_future'], ['primitive_prediction'] as in previous response.
    # This is a condensed representation of that logic:
    velocities_temp = {}
    last_pos = None
    for fid in analysis_df.index: # Iterate through the full frame range
        pos = wrist_coords_dict.get(fid)
        if pos is not None:
            velocities_temp[fid] = np.linalg.norm(pos - last_pos) if last_pos is not None else 0.0
            last_pos = pos
        else: velocities_temp[fid] = np.nan # Or 0, depending on desired handling for missing wrist
    analysis_df['raw_velocity'] = pd.Series(velocities_temp)
    analysis_df['smoothed_velocity'] = analysis_df['raw_velocity'].rolling(window=smoothing_window_velocity, min_periods=1, center=True).mean().fillna(0)

    for fid_idx, fid in enumerate(analysis_df.index):
        # Motion Prediction
        win_coords = [wrist_coords_dict.get(f) for f in range(max(0, fid - displacement_window_size + 1), fid + 1) if wrist_coords_dict.get(f) is not None]
        dist = sum(np.linalg.norm(win_coords[j] - win_coords[j-1]) for j in range(1, len(win_coords))) if len(win_coords) >= 2 else 0.0
        analysis_df.loc[fid, 'motion_prediction_kinematic'] = 1 if dist >= distance_threshold_motion else 0
    analysis_df['motion_prediction_kinematic'] = analysis_df['motion_prediction_kinematic'].fillna(0).astype(int)

    # LLM Contact
    if llm_data_path and os.path.exists(llm_data_path):
        with open(llm_data_path, "r") as f: llm_contact_json = json.load(f)
        llm_contacts_s = pd.Series({int(k): (1 if v else 0) for k,v in llm_contact_json.items()}).reindex(analysis_df.index).fillna(0).astype(int)
        analysis_df['contact_llm'] = llm_contacts_s
    else: analysis_df['contact_llm'] = 0
    
    # Contact Future & Predicted Primitives
    temp_contact_future = []
    temp_predicted_primitives = []
    for i in range(len(analysis_df)):
        motion = analysis_df['motion_prediction_kinematic'].iloc[i]
        contact_n = analysis_df['contact_llm'].iloc[i]
        contact_f = 0
        if i + 1 < len(analysis_df) and analysis_df['contact_llm'].iloc[i+1 : min(len(analysis_df), i+1+contact_lookahead_window)].sum() > 0:
            contact_f = 1
        temp_contact_future.append(contact_f)
        # Primitive logic
        if motion == 1: prim = "reach" if contact_n == 0 and contact_f == 1 else ("reposition" if contact_n == 0 else "transport")
        else: prim = "stabilize" if contact_n == 1 else "idle"
        temp_predicted_primitives.append(prim)
    analysis_df['contact_future'] = temp_contact_future
    analysis_df['primitive_prediction'] = temp_predicted_primitives
    predicted_primitives_condensed_list = LabelUtils.convert_predicted_primitives_to_sequence(analysis_df['primitive_prediction'].tolist())


    # --- Ground Truth Processing ---
    frame_wise_gt_primitives_list = [None] * len(analysis_df) # Initialize for the range of df
    
    USE_GROUND_TRUTH = label_path is not None and os.path.exists(label_path)
    if USE_GROUND_TRUTH:
        gt_handedness = LabelUtils.get_handedness(label_path)
        gt_primitives_condensed_list = LabelUtils.convert_gt_labels_to_primitive_sequence(label_path, hand_to_track.lower())
        
        # Populate frame_wise_gt_primitives_list
        try:
            gt_df = pd.read_csv(label_path)
            # This assumes gt_df 'Frame' column aligns with analysis_df.index
            # A more robust mapping might be needed if GT frames are sparse or on different scale
            gt_prims_temp_dict = {}
            f=0
            for _, row in gt_df.iterrows():
                gt_frame_idx = f
                action = str(row['MarkerNames']).lower()
                prim_gt = None
                is_correct_hand_gt = (gt_handedness == "left" and "l_" in action) or \
                                     (gt_handedness == "right" and "r_" in action) or \
                                     (gt_handedness not in ["left", "right"])
                if is_correct_hand_gt:
                    if "reach" in action: prim_gt = "reach"
                    elif "reposition" in action or "retract" in action: prim_gt = "reposition"
                    elif "transport" in action: prim_gt = "transport"
                    elif "stabilize" in action: prim_gt = "stabilize"
                    elif "idle" in action or "rest" in action: prim_gt = "idle"
                gt_prims_temp_dict[gt_frame_idx] = prim_gt
                f+=1
            
            for i, fid in enumerate(analysis_df.index):
                frame_wise_gt_primitives_list[i] = gt_prims_temp_dict.get(fid)

        except Exception as e:
            print(f"Error processing frame-wise GT primitives for video annotation: {e}")

        # Calculate primitive scores
        if gt_primitives_condensed_list and predicted_primitives_condensed_list:
            primitive_scores_dict = _get_primitives_score(predicted_primitives_condensed_list, gt_primitives_condensed_list)
            print(f"Primitive Scores: {primitive_scores_dict}")
        else:
             print("Not enough data for primitive scoring (predicted or GT sequence missing/empty).")


        # ... (Motion metrics calculations from your original script, if still needed) ...

    # --- Plotting --- (from your original script)
    # ... This part should use the `analysis_df`
    try:
        num_subplots = 3
        fig, axes = plt.subplots(num_subplots, 1, figsize=(18, 6 * num_subplots), sharex=True)
        # Plot 1: Velocity
        axes[0].plot(analysis_df.index, analysis_df['raw_velocity'], label='Raw Velocity', alpha=0.5)
        axes[0].plot(analysis_df.index, analysis_df['smoothed_velocity'], label=f'Smoothed Velocity ({smoothing_window_velocity}-frame avg)')
        axes[0].legend(); axes[0].set_ylabel("Velocity"); axes[0].grid(True)
        axes[0].set_title(f"Patient {selected_patient_id} - Hand {hand_to_track.upper()} Analysis")
        # Plot 2: Motion State
        axes[1].step(analysis_df.index, analysis_df['motion_prediction_kinematic'], label='Kinematic Motion Pred', where='mid')
        axes[1].legend(); axes[1].set_ylabel("Motion (0/1)"); axes[1].grid(True)
        # Plot 3: Primitives
        # (Map primitives to numbers for plotting as in your original script)
        # ... plotting logic for primitives ...
        primitive_map_plot = {prim: i for i, prim in enumerate(LabelUtils.PRIMITIVES)}
        # Add any other primitives that might appear
        all_unique_prims_plot = sorted(list(set(analysis_df['primitive_prediction'].dropna().unique().tolist() + [p for p in gt_primitives_condensed_list if p])))
        current_max_idx_plot = len(primitive_map_plot)
        for p_plot in all_unique_prims_plot:
            if p_plot not in primitive_map_plot: primitive_map_plot[p_plot] = current_max_idx_plot; current_max_idx_plot +=1
        
        analysis_df['primitive_prediction_numeric'] = analysis_df['primitive_prediction'].map(primitive_map_plot).fillna(-1)
        axes[2].step(analysis_df.index, analysis_df['primitive_prediction_numeric'], label='Predicted Primitives', where='mid', color='purple')
        axes[2].set_yticks(list(primitive_map_plot.values()))
        axes[2].set_yticklabels(list(primitive_map_plot.keys()))
        axes[2].legend(); axes[2].set_ylabel("Primitive"); axes[2].grid(True)
        axes[2].set_xlabel("Frame Number")

        fig.tight_layout()
        plt.savefig(output_plot_path, dpi=200)
        print(f"Plot saved to {output_plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Error during plotting: {e}")


    print("\nMotion analysis function finished.")
    return analysis_df, frame_wise_gt_primitives_list, primitive_scores_dict, predicted_primitives_condensed_list, gt_primitives_condensed_list




def run_motion_analysis_legacy(
    selected_patient_id: int,
    hand_to_track: str, # "L" or "R"
    pose_data_path: str,
    label_path: Optional[str], # For ground truth motion (0/1) and primitives
    llm_data_path: Optional[str], # For LLM-based contact flags
    output_plot_path: str,
    smoothing_window_velocity: int,
    displacement_window_size: int,
    distance_threshold_motion: float,
    min_wrist_confidence: float,
    gt_subsample_factor: int,
    contact_lookahead_window: int = 10):
    print(f"\n--- Starting Full Analysis (Motion & Primitives) ---")
    print(f"Patient ID: {selected_patient_id}, Hand: {hand_to_track}")
    print(f"Pose Data: {pose_data_path}")
    print(f"LLM Contact Data: {llm_data_path if llm_data_path else 'Not used'}")
    print(f"Ground Truth (Labels): {label_path if label_path else 'Not used'}")

    # --- Load Pose Keypoint Data ---
    print(f"Loading pose data from {pose_data_path}...")
    try:
        with open(pose_data_path, "r") as f:
            pose_data_all_frames = json.load(f)
        print(f"Loaded data for {len(pose_data_all_frames)} frames from pose JSON.")
    except Exception as e:
        print(f"Error loading or decoding pose JSON: {e}")
        return None, None # Changed from sys.exit(1) to allow calling as function

    # --- Extract Specified Wrist Coordinates ---
    print(f"Extracting '{hand_to_track}' wrist coordinates for patient ID {selected_patient_id}...")

    wrist_coords_dict: Dict[int, Optional[np.ndarray]] = {}
    sorted_frame_id_strings = sorted(pose_data_all_frames.keys(), key=int)

    for frame_id_str in sorted_frame_id_strings:
        frame_data = pose_data_all_frames.get(frame_id_str)
        point_to_store = None
        if frame_data and frame_data.get("patient_id_track") == selected_patient_id:
            wrist_key_smooth = f"{'left' if hand_to_track.lower()=='l' else 'right'}_wrist_smooth"
            wrist_key_conf = f"{'left' if hand_to_track.lower()=='l' else 'right'}_wrist_conf"
            coords = frame_data.get(wrist_key_smooth)
            conf = frame_data.get(wrist_key_conf)
            if coords and isinstance(coords, list) and len(coords) == 2 and \
               conf is not None and conf >= min_wrist_confidence:
                point_to_store = np.array(coords, dtype=float)
        wrist_coords_dict[int(frame_id_str)] = point_to_store
    
    processed_frame_ids_int = sorted(wrist_coords_dict.keys())
    if not processed_frame_ids_int:
        print("Error: No frames found for the selected patient ID. Cannot proceed.")
        return None, None

    valid_coords_count = sum(1 for c in wrist_coords_dict.values() if c is not None)
    print(f"Found {valid_coords_count} valid '{hand_to_track}' wrist coordinates for Patient ID {selected_patient_id} across {len(processed_frame_ids_int)} frames.")
    if valid_coords_count == 0:
        print("Error: No valid wrist coordinates found. Cannot proceed.")
        return

    # --- Calculate Raw & Smoothed Velocity ---
    velocities: Dict[int, float] = {}
    last_valid_pos: Optional[np.ndarray] = None
    for frame_id in processed_frame_ids_int:
        current_pos = wrist_coords_dict.get(frame_id)
        if current_pos is not None:
            if last_valid_pos is not None:
                velocities[frame_id] = np.linalg.norm(current_pos - last_valid_pos)
            else:
                velocities[frame_id] = 0.0
            last_valid_pos = current_pos
        else:
            velocities[frame_id] = np.nan
            
    df = pd.DataFrame.from_dict(velocities, orient='index', columns=['raw_velocity'])
    df.index.name = 'frame'

    if processed_frame_ids_int:
         df = df.reindex(range(min(processed_frame_ids_int), max(processed_frame_ids_int) + 1)).sort_index()
    else: # Handle empty case
        df = pd.DataFrame(columns=['raw_velocity', 'smoothed_velocity', 'motion_prediction_kinematic', 'contact_llm', 'contact_future', 'primitive_prediction'])
        # Return early if no data to process
        print("No processed frames to analyze.")
        return df, []

    # --- Predict Motion based on Window Displacement ---
    motion_prediction_list: List[int] = []
    coord_list_for_windowing = [wrist_coords_dict.get(fid) for fid in processed_frame_ids_int]
    for i in range(len(coord_list_for_windowing)):
        window_start_idx = max(0, i - displacement_window_size + 1)
        current_window_coords_raw = coord_list_for_windowing[window_start_idx : i+1]
        window_coords_valid = [pt for pt in current_window_coords_raw if pt is not None]
        total_distance_in_window = 0.0
        if len(window_coords_valid) >= 2:
            for j in range(1, len(window_coords_valid)):
                total_distance_in_window += np.linalg.norm(window_coords_valid[j] - window_coords_valid[j-1])
        motion_prediction_list.append(1 if total_distance_in_window >= distance_threshold_motion else 0)
    df['motion_prediction_kinematic'] = motion_prediction_list
    
    # --- Load LLM Contact Data ---
    contact_flags_llm = pd.Series(dtype=int) # Initialize as empty Series
    if llm_data_path and os.path.exists(llm_data_path):
        print(f"Loading LLM contact data from {llm_data_path}...")
        try:
            with open(llm_data_path, "r") as f:
                llm_contact_json = json.load(f)
            
            # Convert LLM data (dict of frame_str: bool) to a Series aligned with df.index
            # Ensure keys are integers for proper alignment
            llm_contact_data_int_keys = {int(k): (1 if v else 0) for k, v in llm_contact_json.items()}
            contact_flags_llm = pd.Series(llm_contact_data_int_keys, name='contact_llm').reindex(df.index)
            # Forward fill NaNs that might result from LLM data not covering all frames,
            # or decide on a different strategy (e.g., fill with 0)
            contact_flags_llm = contact_flags_llm.fillna(0).astype(int) # Default to no contact if missing
            df['contact_llm'] = contact_flags_llm
            print(f"LLM contact data loaded and aligned. Found {df['contact_llm'].sum()} contact frames.")
        except Exception as e:
            print(f"Error loading or processing LLM contact data: {e}. Continuing without it.")
            df['contact_llm'] = 0 # Default to no contact if loading fails
    else:
        print("LLM contact data path not provided or file not found. Assuming no contact for primitive prediction.")
        df['contact_llm'] = 0 # Default to no contact

    # --- Predict Primitives based on Motion and LLM Contact ---
    print("Predicting primitives using kinematic motion and LLM contact...")
    predicted_primitives_list: List[str] = []
    # Ensure 'motion_prediction_kinematic' and 'contact_llm' are available in df
    if 'motion_prediction_kinematic' not in df.columns:
        print("Error: 'motion_prediction_kinematic' not found. Cannot predict primitives.")
        df['motion_prediction_kinematic'] = 0 # Fallback
    
    num_frames_for_primitives = len(df)
    for i in range(num_frames_for_primitives):
        current_motion = df['motion_prediction_kinematic'].iloc[i]
        contact_now = df['contact_llm'].iloc[i]
        
        # Determine contact_next: 1 if any contact within the lookahead window, else 0
        contact_next = 0
        window_end_idx = min(num_frames_for_primitives, i + 1 + contact_lookahead_window)
        if i + 1 < num_frames_for_primitives : # only look ahead if not at the end
            if df['contact_llm'].iloc[i+1:window_end_idx].sum() > 0:
                contact_next = 1
        
        primitive = "idle" # Default primitive
        if current_motion == 1:
            if contact_now == 0:
                if contact_next == 1:
                    primitive = "reach"
                else:
                    primitive = "reposition"
            else: # contact_now == 1
                primitive = "transport"
        else: # current_motion == 0
            if contact_now == 1:
                primitive = "stabilize"
            else: # contact_now == 0
                primitive = "idle"
        predicted_primitives_list.append(primitive)
    
    df['primitive_prediction'] = predicted_primitives_list
    print("Primitive prediction complete.")

    # --- Ground Truth Processing (Motion & Primitives) ---
    y_true_motion_aligned: Optional[np.ndarray] = None
    gt_primitive_sequence_condensed: List[str] = []
    motion_metrics_calculated = False
    primitive_metrics_calculated = False

    USE_GROUND_TRUTH = label_path is not None and os.path.exists(label_path)

    if USE_GROUND_TRUTH:
        print(f"Loading and processing ground truth from {label_path}...")
        gt_handedness = LabelUtils.get_handedness(label_path)
        print(f"Determined ground truth handedness for labels: {gt_handedness}")

        # 1. Ground Truth for Motion (0/1)
        try:
            df_labels = pd.read_csv(label_path)
            action_seq_gt_motion_full = []
            # Assuming same motion/no-motion primitive sets as original
            motion_primitives_gt = {"reach", "reposition", "transport", "retract"}
            no_motion_primitives_gt = {"stabilize", "idle", "rest"}

            # Filter labels by hand_to_track if gt_handedness matches, or use all if ambiguous
            # This logic might need refinement based on exact GT structure
            relevant_labels = df_labels
            if gt_handedness == "left" and hand_to_track.lower() == "l":
                 relevant_labels = df_labels[df_labels['MarkerNames'].str.lower().str.contains("l_")]
            elif gt_handedness == "right" and hand_to_track.lower() == "r":
                 relevant_labels = df_labels[df_labels['MarkerNames'].str.lower().str.contains("r_")]
            # If gt_handedness is 'unknown' or doesn't match, this might use all labels or require specific logic.
            # For simplicity, we process all rows if no clear hand match or if gt_handedness is ambiguous.

            for _, row in relevant_labels.iterrows():
                action = str(row['MarkerNames']).lower()
                is_motion = any(prim in action for prim in motion_primitives_gt)
                is_no_motion = any(prim in action for prim in no_motion_primitives_gt)
                if is_motion: action_seq_gt_motion_full.append(1)
                elif is_no_motion: action_seq_gt_motion_full.append(0)
                else: action_seq_gt_motion_full.append(0) # Default for unclassified

            y_true_motion_full_np = np.array(action_seq_gt_motion_full)
            
            # Align motion GT with predictions
            # This part is from your original script, ensure 'motion_prediction_kinematic' is used for preds
            final_motion_preds_np = df['motion_prediction_kinematic'].dropna().to_numpy()

            if len(final_motion_preds_np) > 0 and len(y_true_motion_full_np) > 0 and gt_subsample_factor > 0:
                preds_subsampled = final_motion_preds_np[::gt_subsample_factor]
                gt_subsampled = y_true_motion_full_np[::gt_subsample_factor]
                final_alignment_len = min(len(preds_subsampled), len(gt_subsampled))

                if final_alignment_len > 0:
                    y_pred_motion_aligned = preds_subsampled[:final_alignment_len]
                    y_true_motion_aligned = gt_subsampled[:final_alignment_len]
                    
                    accuracy = accuracy_score(y_true_motion_aligned, y_pred_motion_aligned)
                    precision = precision_score(y_true_motion_aligned, y_pred_motion_aligned, zero_division=0)
                    recall = recall_score(y_true_motion_aligned, y_pred_motion_aligned, zero_division=0)
                    f1 = f1_score(y_true_motion_aligned, y_pred_motion_aligned, zero_division=0)
                    conf_matrix = confusion_matrix(y_true_motion_aligned, y_pred_motion_aligned)

                    print(f"\n--- Motion Metrics (Kinematic Prediction vs GT Labels) ---")
                    print(f"(Subsampled by {gt_subsample_factor}, Thresh={distance_threshold_motion}, Win={displacement_window_size})")
                    print(f"Accuracy:  {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                    print("Confusion Matrix (Rows: True, Cols: Pred):\n", conf_matrix)
                    motion_metrics_calculated = True
                else: print("Motion GT Error: Sequences too short for comparison after subsampling.")
            else: print("Motion GT Error: Ground truth or prediction sequence empty, or invalid subsample factor.")

        except Exception as e:
            print(f"Error during motion ground truth processing or metrics: {e}")

        # 2. Ground Truth for Primitives
        try:
            # Use the hand_to_track for filtering GT primitives
            # If gt_handedness matches hand_to_track, it's good.
            # If gt_handedness is 'unknown', we might assume labels are for the tracked hand or not filter.
            # For now, we use the hand_to_track to guide primitive extraction from GT.
            gt_primitive_sequence_condensed = LabelUtils.convert_gt_labels_to_primitive_sequence(label_path, hand_to_track.lower()) # pass hand_to_track
            
            if gt_primitive_sequence_condensed:
                print(f"Processed GT primitive sequence (condensed): {gt_primitive_sequence_condensed}")
                
                # Get condensed predicted primitive sequence (frame-wise predictions -> sequence)
                predicted_primitive_sequence_condensed = LabelUtils.convert_predicted_primitives_to_sequence(
                    df['primitive_prediction'].tolist() # Use frame-wise predictions
                )
                print(f"Predicted primitive sequence (condensed): {predicted_primitive_sequence_condensed}")

                if predicted_primitive_sequence_condensed:
                    primitive_scores = _get_primitives_score(predicted_primitive_sequence_condensed, gt_primitive_sequence_condensed)
                    print(f"\n--- Primitive Prediction Metrics (Rule-based vs GT Labels) ---")
                    print(f"Edit Score (Normalized %): {primitive_scores['edit_score_normalized']:.2f}")
                    print(f"Action Error Rate: {primitive_scores['action_error_rate']:.4f}")
                    print(f"MAE per Primitive: {json.dumps(primitive_scores['mae_per_primitive'], indent=2)}")
                    primitive_metrics_calculated = True
                else:
                    print("Primitive Prediction Warning: No predicted primitives to compare.")
            else:
                print("Primitive GT Warning: No ground truth primitives extracted or processed.")

        except Exception as e:
            print(f"Error during primitive ground truth processing or metrics: {e}")
    else:
        print("Skipping all ground truth processing and metrics (no label_path or file not found).")


    # --- Plotting ---
    print("Generating plot...")
    # Determine number of subplots needed
    num_subplots = 3 # Velocity, Motion State, Primitives
    fig, axes = plt.subplots(num_subplots, 1, figsize=(18, 6 * num_subplots), sharex=True)
    
    # Plot 1: Velocity
    ax1 = axes[0]
    color_raw_vel = 'lightblue'
    color_smooth_vel = 'dodgerblue'
    ax1.plot(df.index, df['raw_velocity'], label='Raw Velocity (Pixel/Frame)', color=color_raw_vel, alpha=0.5, linewidth=1)
    ax1.plot(df.index, df['smoothed_velocity'], label=f'Smoothed Velocity ({smoothing_window_velocity}-frame avg)', color=color_smooth_vel, linewidth=1.5)
    ax1.set_ylabel("Velocity (Pixel/Frame)", color=color_smooth_vel)
    ax1.tick_params(axis='y', labelcolor=color_smooth_vel)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper left')
    ax1.set_title(f"Patient {selected_patient_id} - '{hand_to_track.upper()}' Wrist Analysis", fontsize=14)

    # Plot 2: Motion State (Kinematic Prediction vs GT)
    ax2 = axes[1]
    color_kinematic_pred = 'orangered'
    color_motion_gt = 'forestgreen'
    
    ax2.step(df.index, df['motion_prediction_kinematic'], label=f'Kinematic Motion Pred (Th={distance_threshold_motion:.1f}px, Win={displacement_window_size}f)', where='mid', color=color_kinematic_pred, linestyle='-', linewidth=2)
    
    if USE_GROUND_TRUTH and motion_metrics_calculated and y_true_motion_aligned is not None:
        # For plotting GT motion, align its indices with df.index based on subsampling
        # This visualization assumes y_true_motion_aligned corresponds to df.index[::gt_subsample_factor]
        gt_motion_plot_indices = df.index[::gt_subsample_factor][:len(y_true_motion_aligned)]
        if len(gt_motion_plot_indices) == len(y_true_motion_aligned):
            ax2.step(gt_motion_plot_indices, y_true_motion_aligned, label=f'GT Motion (Subsampled by {gt_subsample_factor})', where='mid', color=color_motion_gt, linestyle='--', linewidth=2)
        else:
            print(f"Warning: Length mismatch for plotting GT Motion. GT Motion not plotted. Indices: {len(gt_motion_plot_indices)}, Data: {len(y_true_motion_aligned)}")
    
    ax2.set_ylabel('Motion State (0/1)', color='black')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Plot 3: Primitive Prediction (Rule-based vs GT)
    ax3 = axes[2]
    color_primitive_pred = 'purple'
    color_primitive_gt = 'darkcyan'
    
    # Map primitive strings to numerical values for plotting
    primitive_map = {prim: i for i, prim in enumerate(LabelUtils.PRIMITIVES)}
    # Add any other primitives that might appear in predictions or GT if not in PRIMITIVES
    all_unique_primitives = sorted(list(set(df['primitive_prediction'].unique().tolist() + gt_primitive_sequence_condensed)))
    current_max_idx = len(primitive_map)
    for prim in all_unique_primitives:
        if prim not in primitive_map:
            primitive_map[prim] = current_max_idx
            current_max_idx +=1
    
    df['primitive_prediction_numeric'] = df['primitive_prediction'].map(primitive_map).fillna(-1) # Use -1 for unmapped
    
    ax3.step(df.index, df['primitive_prediction_numeric'], label='Predicted Primitives (Rule-based)', where='mid', color=color_primitive_pred, linestyle='-', linewidth=2)

    if USE_GROUND_TRUTH and primitive_metrics_calculated and gt_primitive_sequence_condensed:
        # This is challenging because gt_primitive_sequence_condensed is not frame-wise.
        # We need to reconstruct a frame-wise GT primitive sequence for easy plotting, or plot segments.
        # For simplicity here, we'll just mark regions or plot the sequence as horizontal lines if we had start/end frames.
        # A simpler plot: show the condensed GT sequence alongside.
        # For now, let's try to create a step plot for GT primitives if we can map it.
        # This requires knowing the duration of each GT primitive. The provided utils.py does not give frame indices for GT primitives.
        # We will skip plotting GT primitives directly on the timeline for now due to this complexity.
        # Instead, the metrics are printed.
        # A placeholder:
        print("Note: Plotting of ground truth *primitive sequence* on the timeline is complex due to unknown GT primitive durations. Metrics are provided above.")
        # One could add text annotations or a secondary y-axis with the GT sequence listed if needed.

    ax3.set_ylabel('Predicted Primitive', color='black')
    ax3.set_yticks(list(primitive_map.values()))
    ax3.set_yticklabels(list(primitive_map.keys()))
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.set_xlabel("Frame Number (Original Video Index)")


    fig.tight_layout()
    try:
        plt.savefig(output_plot_path, dpi=200)
        print(f"Plot saved to {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)
    print("\nAnalysis complete.")


# --- Main Analysis Class ---

class VideoAnalysisPipeline:
    def __init__(self,
                 dino_model_id: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 yolo_pose_model_path: str = "yolo11x-pose",
                 output_base_dir: str = "analysis_results",
                 results_log_csv_path: str = "pipeline_summary_results.csv"): # New for CSV logging
        # ... (constructor as before)
        self.dino_model_id = dino_model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.yolo_pose_model_path = yolo_pose_model_path
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)

        self.processor = None
        self.model = None
        self._load_dino_model()

        self.results_log_csv_path = os.path.join(self.output_base_dir, results_log_csv_path)
        self.results_data_list = [] # To hold data before writing all at once, or can write row by row
        # Initialize CSV with headers if it doesn't exist (for row-by-row append)
        self._initialize_results_csv()


    def _initialize_results_csv(self):
        """Initializes the results CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.results_log_csv_path):
            with open(self.results_log_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "video_identifier", "activity", "video_path", "label_path", "hand_tracked",
                    "edit_score_normalized", "action_error_rate",
                    "predicted_primitive_sequence", "gt_primitive_sequence",
                    # Add more MAE scores if needed, or just the average
                    "avg_mae_defined_primitives"
                ])
        else:
            # Check if headers are present, if not, add them
            try:
                with open(self.results_log_csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    if headers is None or headers[0] != "video_identifier": # Basic check
                        # File exists but is empty or has wrong headers, re-initialize
                         with open(self.results_log_csv_path, 'w', newline='') as f_write:
                            writer = csv.writer(f_write)
                            writer.writerow([
                                "video_identifier", "activity", "video_path", "label_path", "hand_tracked",
                                "edit_score_normalized", "action_error_rate",
                                "predicted_primitive_sequence", "gt_primitive_sequence",
                                "avg_mae_defined_primitives"
                            ])
            except Exception as e: # Handle empty file or other read errors
                print(f"Could not verify headers for {self.results_log_csv_path}, re-initializing. Error: {e}")
                with open(self.results_log_csv_path, 'w', newline='') as f_write:
                    writer = csv.writer(f_write)
                    writer.writerow([
                        "video_identifier", "activity", "video_path", "label_path", "hand_tracked",
                        "edit_score_normalized", "action_error_rate",
                        "predicted_primitive_sequence", "gt_primitive_sequence",
                        "avg_mae_defined_primitives"
                    ])


    def _log_video_result(self, result_data: Dict):
        """Appends a single video's result to the CSV log file."""
        # Ensure all expected keys are present, defaulting if necessary
        expected_keys = [
            "video_identifier", "activity", "video_path", "label_path", "hand_tracked",
            "edit_score_normalized", "action_error_rate",
            "predicted_primitive_sequence", "gt_primitive_sequence",
            "avg_mae_defined_primitives"
        ]
        
        row_to_write = []
        for key in expected_keys:
            value = result_data.get(key)
            if isinstance(value, list): # Join lists into a string for CSV
                row_to_write.append(";".join(map(str, value)))
            elif value is None:
                row_to_write.append("N/A")
            else:
                row_to_write.append(value)

        try:
            with open(self.results_log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_to_write)
        except Exception as e:
            print(f"Error writing result to CSV {self.results_log_csv_path}: {e}")






    def _load_dino_model(self):
        """Loads the Grounding DINO model and processor."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.dino_model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_model_id).to(self.device)
            print(f"Grounding DINO model '{self.dino_model_id}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading Grounding DINO model: {e}")
            # Depending on desired behavior, you might want to raise the exception or exit
            raise

    def _detect_objects(self, image: np.ndarray, text_prompt: str) -> Tuple[List, List, List]:
        """Detects objects in an image using the loaded Grounding DINO model."""
        if self.model is None or self.processor is None:
            print("DINO Model or processor not available for detection.")
            return [], [], []

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]]).to(self.device) # (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes
        )
        if not results or not results[0]: return [], [], []
        detections = results[0]
        return detections["boxes"].cpu().tolist(), detections["labels"], detections["scores"].cpu().tolist()

    def _annotate_frame_specific(self, image: np.ndarray, patient_boxes: List, hand_boxes: List) -> np.ndarray:
        """Annotates a frame with patient and hand boxes."""
        annotated_img = image.copy()
        for box in patient_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for patient
            cv2.putText(annotated_img, "Patient", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        for box in hand_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for hand
            cv2.putText(annotated_img, "Hand", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        return annotated_img

    def run_initial_object_detection(self, video_path: str, video_identifier: str, frame_stride: int = 1,
                                      skipped_frame_handling: str = "passthrough") -> Tuple[Optional[str], Optional[str]]:
        """
        Runs initial object (person, hand) detection on a video.

        Args:
            video_path (str): Path to the input video.
            video_identifier (str): A unique name for the video.
            frame_stride (int): Process every Nth frame.
            skipped_frame_handling (str): "passthrough" or "skip".

        Returns:
            Tuple[Optional[str], Optional[str]]: Paths to the output video and JSON data, or None if failed.
        """
        if not os.path.exists(video_path):
            print(f"Error: Input video not found at {video_path}")
            return None, None

        video_output_dir = os.path.join(self.output_base_dir, video_identifier)
        os.makedirs(video_output_dir, exist_ok=True)

        output_video_path = os.path.join(video_output_dir, f"{video_identifier}_dino_output.mp4")
        json_data_path = os.path.join(video_output_dir, f"{video_identifier}_video_data.json")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_stride if frame_stride > 0 else fps, (W, H)) # Adjust FPS for stride

        print(f"Processing video for DINO detection: {video_path}")
        all_frames_data = {}

        try:
            for frame_idx in tqdm(range(total_frames), desc=f"DINO Detection for {video_identifier}"):
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % frame_stride == 0:
                    person_boxes, _, person_scores = self._detect_objects(frame, "person.")
                    hand_boxes, _, hand_scores = self._detect_objects(frame, "hand.")

                    current_patient_boxes_for_json = [{"box": b, "score": s} for b, s in zip(person_boxes, person_scores)]
                    current_hand_boxes_for_json = [{"box": b, "score": s} for b, s in zip(hand_boxes, hand_scores)]

                    annotated_frame = self._annotate_frame_specific(frame, person_boxes, hand_boxes)
                    out_writer.write(annotated_frame)

                    all_frames_data[frame_idx] = {
                        "patient_boxes": current_patient_boxes_for_json,
                        "hand_boxes": current_hand_boxes_for_json
                    }
                elif skipped_frame_handling == "passthrough":
                    out_writer.write(frame)
        finally:
            cap.release()
            out_writer.release()

        with open(json_data_path, "w") as f:
            json.dump(all_frames_data, f, indent=4)
        print(f"DINO detection finished. Video: {output_video_path}, Data: {json_data_path}")
        return output_video_path, json_data_path

    def generate_patient_ids(self, video_data_json_path: str, video_identifier: str) -> Optional[str]:
        """
        Generates patient IDs from detection data.
        (This is a placeholder for your `pid_generator` logic)
        """
        # Placeholder: Assumes pid_generator is available and imported
        # from patient_id_generator import pid_generator # Should be at top level
        try:
            output_pid_json_path = os.path.join(self.output_base_dir, video_identifier, f"{video_identifier}_pid_data.json")
            # This is where you'd call your actual pid_generator function
            tracked_data = pid_generator(video_data_json_path, output_pid_json_path)
            # For now, let's assume it creates the file. If it returns data, process it.
            if not os.path.exists(video_data_json_path): # Added check
                 print(f"Error: video_data_json_path not found for PID generation: {video_data_json_path}")
                 return None
            print(f"Running patient ID generation. Input: {video_data_json_path}, Output: {output_pid_json_path}")
            if 'pid_generator' not in globals(): # If not imported
                print("Warning: pid_generator not found. Creating a dummy pid_data.json by copying video_data.json for flow.")
                shutil.copy(video_data_json_path, output_pid_json_path)
            else:
                pid_generator(video_data_json_path, output_pid_json_path) # Call actual

            return output_pid_json_path
        except Exception as e:
            print(f"Error in patient ID generation for {video_identifier}: {e}")
            return None


    def select_patient_and_hands(self, video_path: str, pid_json_path: str, video_identifier: str,
                                 pre_selected_patient_id_int: Optional[int] = None,
                                 pre_selected_hands: Optional[str] = None, # "L", "R", or "BOTH"
                                 auto_hand_choice_from_label: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]: # (patient_track_id, hands_to_track)
        """
        Allows selection of a patient and hands to track.
        Can be interactive or use pre-selected values.
        (Incorporates logic for cropping and displaying for selection if not pre-selected)
        """
        if not pid_json_path or not os.path.exists(pid_json_path):
            print(f"Error: PID JSON path not found: {pid_json_path}")
            return None, None
        with open(pid_json_path, 'r') as f:
            tracked_data = json.load(f) # Assuming pid_json_path contains the direct output of pid_generator

        if not tracked_data:
            print(f"No tracking data found in {pid_json_path}")
            return None, None

        if pre_selected_patient_id_int is not None and pre_selected_hands is not None:
            print(f"Using pre-selected patient ID: {pre_selected_patient_id_int} and hands: {pre_selected_hands}")
            return pre_selected_patient_id_int, pre_selected_hands.upper()

        # --- Logic for interactive selection (simplified from script) ---
        # This part would involve GUI or more complex CLI if not pre-selecting
        first_occurrences = {}
        sorted_frame_keys = sorted(tracked_data.keys(), key=lambda x: int(x))
        for frame_num_str in sorted_frame_keys:
            frame_info = tracked_data[frame_num_str]
            for track_id_int_str, track_details in frame_info.items(): # Assuming track_id might be string key
                track_id_int = int(track_id_int_str)
                if track_id_int not in first_occurrences:
                    # Basic check, your pid_generator output structure might differ
                    if "box" in track_details and "id_str" in track_details :
                         first_occurrences[track_id_int] = {
                            "frame_idx": int(frame_num_str),
                            "box": track_details["box"],
                            "id_str": track_details["id_str"],
                            "score": track_details.get("score")
                        }

        if not first_occurrences:
            print("No patients found in tracking data for selection.")
            return None, None

        # Simplified selection: take the first available patient ID if not pre-selected
        # In a real app, implement the crop display and input() logic here
        if not first_occurrences:
            print("No patients identified for selection.")
            return None, None

        # For non-interactive mode if not pre-selected:
        # Fallback: select the patient track with the most occurrences or first one
        # This is a simplification. The original script had a crop-based visual selection.
        selected_patient_track_id_int = list(first_occurrences.keys())[0]
        selected_patient_id_str = first_occurrences[selected_patient_track_id_int]['id_str']
        print(f"Auto-selecting patient: {selected_patient_id_str} (Internal ID: {selected_patient_track_id_int}) due to non-interactive mode.")

        hands_to_track = "BOTH" # Default
        if auto_hand_choice_from_label and auto_hand_choice_from_label.upper() in ["L", "R", "BOTH"]:
            hands_to_track = auto_hand_choice_from_label.upper()
        elif pre_selected_hands:
             hands_to_track = pre_selected_hands.upper()

        print(f"Selected patient: {selected_patient_track_id_int}, Hands: {hands_to_track}")
        return selected_patient_track_id_int, hands_to_track


    def extract_poses(self, video_path: str, pid_json_path: str, video_identifier: str,
                      selected_patient_id_int: int, hand_to_track: str,
                      jump_threshold: float = 50.0, smooth_window: int = 10,
                      min_keypoint_confidence: float = 0.3) -> Optional[str]:
        """
        Extracts pose keypoints for the selected patient.
        (Placeholder for `extract_pose_keypoints` logic)
        """
        # from pose_extractor import extract_pose_keypoints # Ensure imported
        try:
            output_pose_json_path = os.path.join(self.output_base_dir, video_identifier, "keypoints_enhanced_output.json")
            output_pose_video_path = os.path.join(self.output_base_dir, video_identifier, "video_enhanced_pose_output.mp4")

            print(f"Running pose extraction for patient {selected_patient_id_int}, hand {hand_to_track}")
            if 'extract_pose_keypoints' not in globals():
                print("Warning: extract_pose_keypoints not found. Cannot extract poses.")
                return None # Or handle appropriately
            else:
                success = extract_pose_keypoints(
                    tracker_hand=hand_to_track.upper(),
                    pid_json_path=pid_json_path,
                    output_json_path=output_pose_json_path,
                    output_video_path=output_pose_video_path,
                    jump_threshold=jump_threshold,
                    smooth_window=smooth_window,
                    patient_id_track=selected_patient_id_int,
                    video_path=video_path,
                    min_keypoint_confidence=min_keypoint_confidence,
                    yolo_pose_model_path=self.yolo_pose_model_path # Pass from class
                )
                if not success:
                    print(f"Pose extraction failed for {video_identifier}.")
                    return None
            return output_pose_json_path
        except Exception as e:
            print(f"Error in pose extraction for {video_identifier}: {e}")
            return None

    def map_hands_to_patient(self, video_path: str, pose_data_json_path: str, hand_detections_pid_json_path: str,
                             video_identifier: str, selected_patient_id_int: int, hand_to_track: str) -> Optional[str]:
        """
        Maps detected hands to the selected patient using pose data.
        (Placeholder for `map_hand_to_patient_enhanced` logic)
        """
        # from hand_id_generator import map_hand_to_patient_enhanced # Ensure imported
        try:
            output_dir = os.path.join(self.output_base_dir, video_identifier)
            output_hand_json = os.path.join(output_dir, f"patient_{selected_patient_id_int}_hand_{hand_to_track}_mapped_data.json")
            output_hand_video = os.path.join(output_dir, f"patient_{selected_patient_id_int}_hand_{hand_to_track}_tracked_video.mp4")

            print(f"Running hand mapping for patient {selected_patient_id_int}, hand {hand_to_track}.")
            if 'map_hand_to_patient_enhanced' not in globals():
                print("Warning: map_hand_to_patient_enhanced not found. Cannot map hands.")
                # with open(output_hand_json, 'w') as f: json.dump({}, f)
                return None
            else:
                map_hand_to_patient_enhanced(
                    selected_patient_id=selected_patient_id_int,
                    hand_to_track=hand_to_track.upper(),
                    video_path=video_path,
                    pose_data_path=pose_data_json_path,
                    hand_detections_path=hand_detections_pid_json_path, # This should be pid_json_path
                    output_json_path=output_hand_json,
                    output_video_path=output_hand_video,
                )
            return output_hand_json
        except Exception as e:
            print(f"Error in hand mapping for {video_identifier}: {e}")
            return None

    def analyze_llm_interactions(self, video_path: str, mapped_hand_json_path: str, video_identifier: str,
                                 selected_patient_id_int: int, hand_to_track: str,
                                 llm_model_name_for_analysis: Optional[str] = "llava-hf/llava-onevision-qwen2-7b-ov-hf", # If your llm_caller uses it
                                 objects_for_prompt: str = "",
                                 wrist_roi_padding: int = 30,
                                 min_wrist_confidence_for_roi: float = 0.3,
                                 save_debug_frames: bool = True,
                                 debug_frame_interval: int = 25) -> Optional[str]:
        """
        Analyzes interactions using an LLM.
        (Placeholder for `llm_interaction_analyzer` logic)
        """
        # from llm_caller import llm_interaction_analyzer # Ensure imported
        try:
            output_dir = os.path.join(self.output_base_dir, video_identifier)
            output_llm_video = os.path.join(output_dir, f"patient_{selected_patient_id_int}_hand_{hand_to_track}_llm_interactions.mp4")
            output_llm_interactions_json = os.path.join(output_dir, f"patient_{selected_patient_id_int}_hand_{hand_to_track}_llm_log.json") # This is key output
            output_llm_roi_json = os.path.join(output_dir, f"final_llm_wrist_roi_pat{selected_patient_id_int}_{hand_to_track}_roi_details.json")

            print(f"Running LLM interaction analysis for patient {selected_patient_id_int}, hand {hand_to_track} (mock).")
            if 'llm_interaction_analyzer' not in globals():
                print("Warning: llm_interaction_analyzer not found. Cannot analyze LLM interactions.")
                # with open(output_llm_interactions_json, 'w') as f: json.dump({}, f) # Dummy for flow
                return None
            else:
                llm_interaction_analyzer(
                    selected_patient_id=selected_patient_id_int,
                    hand_to_track=hand_to_track.upper(),
                    video_path=video_path,
                    mapped_hand_data_path=mapped_hand_json_path,
                    output_video_path=output_llm_video,
                    output_roi_details_json=output_llm_roi_json,
                    output_interactions_json=output_llm_interactions_json,
                    objects_for_prompt=objects_for_prompt,
                    wrist_roi_padding=wrist_roi_padding,
                    min_wrist_confidence_for_roi=min_wrist_confidence_for_roi,
                    save_debug_frames=save_debug_frames,
                    debug_frame_interval=debug_frame_interval,
                    llm_model_name=llm_model_name_for_analysis # If your function accepts it
                )
            return output_llm_interactions_json
        except Exception as e:
            print(f"Error in LLM interaction analysis for {video_identifier}: {e}")
            return None



    def run_full_motion_analysis(self, pose_data_json_path: str,
                                 video_identifier: str, selected_patient_id_int: int, hand_to_track: str,
                                 label_path: Optional[str], # For GT
                                 llm_contact_json_path: Optional[str], # From LLM analysis
                                 # Motion analysis parameters:
                                 smoothing_window_velocity: int = 10,
                                 displacement_window_size: int = 10,
                                 distance_threshold_motion: float = 5.0,
                                 min_wrist_confidence_motion: float = 0.3,
                                 gt_subsample_factor: int = 1,
                                 contact_lookahead_window: int = 10
                                 ) -> Optional[str]:
        """
        Performs the final motion and primitive analysis.
        (Corresponds to `run_motion_analysis` from the script)
        """
        # This method would encapsulate the logic from your `run_motion_analysis` function.
        # For brevity, I'm not re-implementing the entire function here, but it would:
        # 1. Load pose data, LLM contact data, and GT labels (if provided).
        # 2. Calculate velocities, predict motion kinematically.
        # 3. Predict primitives based on motion and LLM contact.
        # 4. If GT labels are available, calculate metrics for motion and primitives.
        # 5. Generate and save plots.

        print(f"\n--- Starting Full Motion Analysis for {video_identifier} ---")
        output_dir = os.path.join(self.output_base_dir, video_identifier)
        output_plot_path = os.path.join(output_dir, f"patient_{selected_patient_id_int}_{hand_to_track.lower()}_full_analysis.png")

        # ---- Actual call to your run_motion_analysis logic ----
        # This assumes 'run_motion_analysis' is imported or defined.
        # from new_primi import run_motion_analysis # Or wherever it's defined.
        if 'run_motion_analysis' not in globals():
            print("Warning: run_motion_analysis function not found. Skipping full motion analysis.")
            return None, None, None, None, None
        else:
            try:
                # Ensure all paths are valid before calling
                if not os.path.exists(pose_data_json_path):
                    print(f"Error: Pose data path for motion analysis not found: {pose_data_json_path}")
                    return None, None, None, None, None
                if llm_contact_json_path and not os.path.exists(llm_contact_json_path):
                    print(f"Warning: LLM contact data path not found: {llm_contact_json_path}. Proceeding without it for primitives.")
                    llm_contact_json_path = None # Treat as not provided
                if label_path and not os.path.exists(label_path):
                    print(f"Warning: GT Label path not found: {label_path}. Proceeding without GT.")
                    label_path = None


                analysis_df_res, frame_wise_gt_res, scores_res, pred_seq_res, gt_seq_res = run_motion_analysis(
                    selected_patient_id=selected_patient_id_int,
                    hand_to_track=hand_to_track.upper(),
                    pose_data_path=pose_data_json_path,
                    label_path=label_path,
                    llm_data_path=llm_contact_json_path,
                    output_plot_path=output_plot_path,
                    smoothing_window_velocity=smoothing_window_velocity,
                    displacement_window_size=displacement_window_size,
                    distance_threshold_motion=distance_threshold_motion,
                    min_wrist_confidence=min_wrist_confidence_motion,
                    gt_subsample_factor=gt_subsample_factor,
                    contact_lookahead_window=contact_lookahead_window
                )
                if analysis_df_res is not None:
                    print(f"Motion analysis and plotting complete. Plot: {output_plot_path}")
                else:
                    print(f"Motion analysis function returned no DataFrame for {video_identifier}.")
                return output_plot_path, analysis_df_res, frame_wise_gt_res, scores_res, pred_seq_res, gt_seq_res

            except Exception as e:
                print(f"Error during pipeline's call to full motion analysis for {video_identifier}: {e}")
                import traceback; traceback.print_exc()
                return None, None, None, None, None, None
        # ---- End call ----


    def generate_summary_annotated_video(self,
                                         original_video_path: str,
                                         analysis_df: pd.DataFrame,
                                         frame_wise_gt_primitives: Optional[List[Optional[str]]],
                                         video_identifier: str,
                                         output_filename_suffix: str = "_summary_annotated.mp4"):
        if analysis_df is None or analysis_df.empty:
            print(f"Analysis DataFrame is empty for {video_identifier}. Skipping summary video generation.")
            return

        if not os.path.exists(original_video_path):
            print(f"Original video not found: {original_video_path}. Skipping summary video generation.")
            return

        output_dir = os.path.join(self.output_base_dir, video_identifier)
        os.makedirs(output_dir, exist_ok=True)
        annotated_video_path = os.path.join(output_dir, f"{video_identifier}{output_filename_suffix}")

        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            print(f"Error opening original video: {original_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        print(f"Generating summary annotated video for {video_identifier} to {annotated_video_path}...")

        # Ensure analysis_df index is suitable for direct lookup or use .iloc if it's a simple range
        # If analysis_df.index corresponds to original video frame numbers:
        df_min_frame = analysis_df.index.min() if not analysis_df.empty else 0
        df_max_frame = analysis_df.index.max() if not analysis_df.empty else -1


        for frame_num in tqdm(range(total_frames_video), desc="Annotating summary video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Get data for the current frame
            # Ensure frame_num is within the bounds of analysis_df and frame_wise_gt_primitives
            motion_val, contact_now_val, contact_future_val, pred_prim_val, gt_prim_val = "N/A", "N/A", "N/A", "N/A", "N/A"

            if frame_num >= df_min_frame and frame_num <= df_max_frame:
                if frame_num in analysis_df.index:
                    row = analysis_df.loc[frame_num]
                    motion_val = str(row.get('motion_prediction_kinematic', "N/A"))
                    contact_now_val = str(row.get('contact_llm', "N/A"))
                    contact_future_val = str(row.get('contact_future', "N/A")) # From the new column
                    pred_prim_val = str(row.get('primitive_prediction', "N/A"))
                # else:
                    # print(f"Frame {frame_num} not in analysis_df index. Min: {df_min_frame}, Max: {df_max_frame}")


            if frame_wise_gt_primitives and 0 <= frame_num < len(frame_wise_gt_primitives):
                gt_prim_val = frame_wise_gt_primitives[frame_num] if frame_wise_gt_primitives[frame_num] is not None else "N/A"
            # else:
                # if not frame_wise_gt_primitives: print(f"No GT primitive list for frame {frame_num}")
                # elif not (0 <= frame_num < len(frame_wise_gt_primitives)): print(f"Frame {frame_num} out of bounds for GT list (len: {len(frame_wise_gt_primitives)})")


            # Prepare text for annotation
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255) # White
            bg_color = (0,0,0) # Black background for text
            line_type = 2
            y_offset = 30
            x_offset = 10

            texts = [
                f"Frame: {frame_num}",
                f"Motion: {motion_val}",
                f"Contact Now: {contact_now_val}",
                f"Contact Future: {contact_future_val}",
                f"Prediction: {pred_prim_val}",
                f"Ground Truth: {gt_prim_val}"
            ]

            for i, text in enumerate(texts):
                text_y = y_offset + i * 25
                # Simple background rectangle
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_type)
                cv2.rectangle(frame, (x_offset -5, text_y - text_height -2), (x_offset + text_width + 5, text_y + 5), bg_color, -1)
                cv2.putText(frame, text, (x_offset, text_y), font, font_scale, font_color, line_type)

            out_writer.write(frame)

        cap.release()
        out_writer.release()
        print(f"Finished generating summary annotated video: {annotated_video_path}")


    def process_video_entry(self,
                            video_identifier: str,
                            video_path: str,
                            label_path: Optional[str],
                            activity_name: str, # NEW: To get activity for logging
                            llm_objects_prompt: str, # NEW: Passed from dynamic setup
                            pre_selected_patient_id: Optional[int] = None,
                            pre_selected_hands_str: Optional[str] = None,
                            frame_stride_dino: int = 1,
                            motion_distance_thresh: float = 5.0,
                            gt_subsample: int = 1,
                            contact_lookahead_window_param: int = 10):
        """
        Processes a single video entry through the entire pipeline.
        """
        print(f"\n\n===== Processing Video Entry: {video_identifier} =====")
        print(f"Video Path: {video_path}")
        print(f"Label Path: {label_path if label_path else 'N/A'}")

        # Stage 1: Initial Object Detection (DINO)
        _, dino_json_path = self.run_initial_object_detection(video_path, video_identifier, frame_stride=frame_stride_dino)
        if not dino_json_path:
            print(f"Failed at DINO detection for {video_identifier}. Skipping further processing.")
            return

        # Stage 2: Patient ID Generation
        pid_json_path = self.generate_patient_ids(dino_json_path, video_identifier)
        if not pid_json_path:
            # Attempt to use dino_json_path as a fallback if pid_generator failed but we need its structure for hand detection
            # This behavior depends on how map_hand_to_patient_enhanced uses hand_detections_path
            print(f"Patient ID generation failed for {video_identifier}. Attempting to use DINO output for hand mapping if structure is compatible.")
            # If `map_hand_to_patient_enhanced` *requires* specific PID processing, this won't work well.
            # For now, we assume it might use general detections if PID-specific ones aren't there.
            # A robust solution would handle this more gracefully based on actual dependencies.
            # pid_json_path = dino_json_path # Risky fallback, use with caution.
            # For now, let's be safer and stop if critical step fails
            print(f"Failed at Patient ID generation for {video_identifier}. Skipping further processing.")
            return


        # Stage 3: Select Patient and Hands
        auto_hand = None
        if label_path and os.path.exists(label_path): # Try to get handedness from GT labels
            try:
                # Using the class method for get_handedness
                gt_hand = LabelUtils.get_handedness(label_path)
                if gt_hand == "left": auto_hand = "L"
                elif gt_hand == "right": auto_hand = "R"
                else: auto_hand = "BOTH" # Or some other default if ambiguous
                print(f"Auto-detected hand for tracking from labels ({label_path}): {auto_hand} (based on {gt_hand})")
            except Exception as e:
                print(f"Could not auto-detect hand from label file {label_path}: {e}")
                auto_hand = "BOTH" # Fallback

        elif pre_selected_hands_str:
            auto_hand = pre_selected_hands_str
        else:
            auto_hand = "BOTH" # Default if no label or pre-selection for hands

        patient_id_to_track, hands_to_track_str = self.select_patient_and_hands(
            video_path, pid_json_path, video_identifier,
            pre_selected_patient_id_int=pre_selected_patient_id,
            auto_hand_choice_from_label=auto_hand # Pass detected/default hand
        )
        if patient_id_to_track is None or hands_to_track_str is None:
            print(f"Failed at patient/hand selection for {video_identifier}. Skipping.")
            return

        # Stage 4: Pose Extraction
        pose_json_path = self.extract_poses(video_path, pid_json_path, video_identifier,
                                            patient_id_to_track, hands_to_track_str,
                                            min_keypoint_confidence=0.3) # Example param
        if not pose_json_path:
            print(f"Failed at pose extraction for {video_identifier}. Skipping.")
            return

        # Stage 5: Map Hands to Patient
        # The hand_detections_pid_json_path for map_hand_to_patient_enhanced was originally the pid_json_path
        mapped_hand_data_json_path = self.map_hands_to_patient(video_path, pose_json_path, pid_json_path,
                                                               video_identifier, patient_id_to_track, hands_to_track_str)
        if not mapped_hand_data_json_path:
            print(f"Failed at hand mapping for {video_identifier}. Skipping.")
            return

        # Stage 6: LLM Interaction Analysis
        llm_log_json_path = self.analyze_llm_interactions(video_path, mapped_hand_data_json_path, video_identifier,
                                                          patient_id_to_track, hands_to_track_str,
                                                          objects_for_prompt=llm_objects_prompt)
        # llm_log_json_path can be None if LLM analysis is optional or fails,
        # motion analysis might proceed without it (current run_motion_analysis handles None llm_data_path)

        # Stage 7: Full Motion Analysis

        plot_path, analysis_df_res, frame_wise_gt_res, scores_res, pred_seq_res, gt_seq_res = self.run_full_motion_analysis(
            pose_data_json_path=pose_json_path,
            video_identifier=video_identifier,
            selected_patient_id_int=patient_id_to_track,
            hand_to_track=hands_to_track_str,
            label_path=label_path,
            llm_contact_json_path=llm_log_json_path,
            distance_threshold_motion=motion_distance_thresh,
            gt_subsample_factor=gt_subsample,
            contact_lookahead_window=contact_lookahead_window_param,
            # Pass other necessary params like smoothing_window_velocity, etc.
            smoothing_window_velocity=10, # Example, make configurable
            displacement_window_size=10,  # Example
            min_wrist_confidence_motion=0.3 # Example
        )

        # Stage 8: Generate Summary Annotated Video
        if analysis_df_res is not None:
            self.generate_summary_annotated_video(
                original_video_path=video_path,
                analysis_df=analysis_df_res,
                frame_wise_gt_primitives=frame_wise_gt_res,
                video_identifier=video_identifier
            )

        # Stage 9: Log Results to CSV (NEW)
        if scores_res: # Only log if scores were computed
            result_log_entry = {
                "video_identifier": video_identifier,
                "activity": activity_name,
                "video_path": video_path,
                "label_path": label_path if label_path else "N/A",
                "hand_tracked": hands_to_track_str,
                "edit_score_normalized": scores_res.get("edit_score_normalized", "N/A"),
                "action_error_rate": scores_res.get("action_error_rate", "N/A"),
                "predicted_primitive_sequence": pred_seq_res if pred_seq_res else [],
                "gt_primitive_sequence": gt_seq_res if gt_seq_res else [],
                "avg_mae_defined_primitives": scores_res.get("mae_per_primitive", {}).get("avg_mae_defined_primitives", "N/A")
            }
            self._log_video_result(result_log_entry)
        else:
            # Log basic info even if scoring failed
            result_log_entry = {
                "video_identifier": video_identifier, "activity": activity_name,
                "video_path": video_path, "label_path": label_path if label_path else "N/A",
                "hand_tracked": hands_to_track_str,
                "edit_score_normalized": "N/A", "action_error_rate": "N/A",
                "predicted_primitive_sequence": pred_seq_res if pred_seq_res else [], # Still log predicted sequence
                "gt_primitive_sequence": "N/A", "avg_mae_defined_primitives": "N/A"
            }
            self._log_video_result(result_log_entry)


        print(f"===== Finished processing for: {video_identifier} =====")


    def process_videos(self,
                       video_label_dict: Dict[str, Dict[str, Optional[str]]],
                       default_patient_id: Optional[int] = None,
                       default_hands_str: Optional[str] = None,
                       global_llm_objects_prompt: str = "related objects",
                       global_motion_threshold: float = 5.0,
                       global_gt_subsample: int = 1,
                       global_dino_frame_stride: int = 1,
                       global_contact_lookahead: int = 10
                      ):
        """
        Processes a dictionary of videos through the pipeline.

        Args:
            video_label_dict (Dict[str, Dict[str, Optional[str]]]):
                A dictionary where keys are video identifiers (e.g., "C00022_deodrant4_2")
                and values are dictionaries like:
                {
                    "video_path": "/path/to/video.mkv",
                    "label_path": "/path/to/labels.csv" (Optional)
                    "pre_selected_patient_id": 1 (Optional)
                    "pre_selected_hands": "R" (Optional)
                    "llm_prompt_objects": "tools on table" (Optional, overrides global)
                    ... other specific params ...
                }
        """
        if not self.model or not self.processor:
            print("DINO Model not loaded. Cannot process videos.")
            return

        for video_id, paths_and_params in video_label_dict.items():
            video_file = paths_and_params.get("video_path")
            label_file = paths_and_params.get("label_path") # Can be None
            activity = paths_and_params.get("activity", "Unknown") # Get activity
            llm_prompt = paths_and_params.get("llm_prompt_objects", "related objects") # Get specific LLM prompt


            if not video_file or not os.path.exists(video_file):
                print(f"Video path for '{video_id}' is missing or invalid: {video_file}. Skipping.")
                continue

            # Use specific params if provided, else use defaults or global overrides
            patient_id = paths_and_params.get("pre_selected_patient_id", default_patient_id)
            hands_str = paths_and_params.get("pre_selected_hands", default_hands_str)
            llm_prompt = paths_and_params.get("llm_prompt_objects", global_llm_objects_prompt)
            motion_thresh = paths_and_params.get("motion_threshold", global_motion_threshold)
            gt_sub = paths_and_params.get("gt_subsample", global_gt_subsample)
            dino_stride = paths_and_params.get("dino_frame_stride", global_dino_frame_stride)
            contact_lookahead = paths_and_params.get("contact_lookahead_window", global_contact_lookahead)


            self.process_video_entry(
                video_identifier=video_id,
                video_path=video_file,
                label_path=label_file,
                activity_name=activity, # Pass activity
                llm_objects_prompt=llm_prompt, # Pass specific LLM prompt
                pre_selected_patient_id=patient_id,
                pre_selected_hands_str=hands_str,
                frame_stride_dino=dino_stride,
                motion_distance_thresh=motion_thresh,
                gt_subsample=gt_sub,
                contact_lookahead_window_param=contact_lookahead
            )

# Example Usage (Import your actual helper functions):
if __name__ == '__main__':
    # --- Configuration for the pipeline ---
    BASE_DATA_PATH = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/" # Example
    # Or for a local test setup:
    # BASE_DATA_PATH = "./dummy_strokerehab_data/" # Ensure this directory exists and has subfolders

    METADATA_CSV_PATH = os.path.join("cleaned_metadata.csv")
    ACTIVITY_YAML_PATH = os.path.join("activities_ground_truth.yaml")

    # --- Load Activity Configuration ---
    activity_config = load_activity_config_from_yaml(ACTIVITY_YAML_PATH)
    if not activity_config:
        print("Warning: Activity configuration is empty. LLM prompts may be generic.")

    # --- Prepare `videos_to_process` from Metadata CSV ---
    videos_to_process = {}
    if os.path.exists(METADATA_CSV_PATH):
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
        # Filter for specific videos if needed, e.g., metadata_df = metadata_df[metadata_df['id'] == "S00027_deodrant_test"]
        metadata_df = metadata_df.head(10)  # Take first 10 rows
        for index, row in metadata_df.iterrows():
            video_id = row['id']
            # Construct full paths, assuming 'path_v' and 'path_l' are relative to a subfolder within BASE_DATA_PATH
            # Example: BASE_DATA_PATH includes up to ".../__data/"
            # Then path_v is like "VideoData/rawVideosADLsandFM/S00027/S00027_feeding1_1.avi"
            # So, join BASE_DATA_PATH with these relative paths.
            # IMPORTANT: Adjust this path construction based on your actual directory layout
            # The metadata CSV seems to have paths like "S00027/S00027_feeding1_1.avi"
            # If your BASE_DATA_PATH is "/gpfs/data/schambralab/quantitativeRehabilitation/__data/",
            # then you need to add the intermediate folders like "VideoData/rawVideosADLsandFM/"
            full_video_path = os.path.join(BASE_DATA_PATH, "VideoData/rawVideosADLsandFM", row['path_v'])
            full_label_path = os.path.join(BASE_DATA_PATH, "rawVideoLabels", row['path_l']) if pd.notna(row['path_l']) else None

            activity_key = row['activity'].lower()
            llm_prompt_for_activity = activity_config.get(activity_key, "related objects") # Default if activity not in YAML

            try:
                hands_to_track = LabelUtils.get_handedness(full_label_path)[0].upper()
            except:
                hands_to_track = "R"
                print(f"Warning: No labels found for {video_id}. Using default hand: {hands_to_track}")

            videos_to_process[video_id] = {
                "video_path": full_video_path,
                "label_path": full_label_path,
                "activity": row['activity'],
                "llm_prompt_objects": llm_prompt_for_activity,
                "pre_selected_patient_id": 0, # Example: always take first detected for automation
                "pre_selected_hands": hands_to_track, # Or determine from labels if possible
                "motion_threshold": 5.0,
                "dino_frame_stride": 5, # Process fewer frames for faster test
                "contact_lookahead_window": 10
            }
    else:
        print(f"Metadata CSV not found at {METADATA_CSV_PATH}. Cannot populate videos_to_process automatically.")
        # Fallback to manual definition if needed for testing a single video
        # videos_to_process = { ... manually define one entry ... }

    num_videos_to_process_subset = 5
    items_to_slice = list(videos_to_process.items())
    actual_num_to_process = min(num_videos_to_process_subset, len(items_to_slice))
        
    videos_to_process_subset_dict = dict(items_to_slice[:actual_num_to_process])

    pipeline = VideoAnalysisPipeline(
        dino_model_id="IDEA-Research/grounding-dino-base", # or other DINO model
        output_base_dir="my_video_analysis_output"
    )

    pipeline.process_videos(videos_to_process_subset_dict)

    print("\n\nPipeline run finished. Check 'my_video_analysis_output' directory.")