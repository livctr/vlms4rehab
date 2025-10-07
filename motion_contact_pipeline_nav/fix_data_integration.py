#!/usr/bin/env python3
"""
Fix data integration between window-based contact data and frame-based motion data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def align_contact_with_motion(contact_df, motion_df):
    """Align window-based contact data with frame-based motion data."""
    print("🔄 Aligning contact data with motion data...")
    
    # Get motion data
    motion_times = motion_df['time_s'].values
    motion_predictions = motion_df['prediction'].values
    
    # Create aligned contact predictions
    contact_predictions = np.zeros(len(motion_times), dtype=int)
    contact_confidences = np.zeros(len(motion_times), dtype=float)
    
    # For each motion frame, find the corresponding contact window
    for i, motion_time in enumerate(motion_times):
        # Find the contact window that contains this motion time
        for _, contact_row in contact_df.iterrows():
            start_time = contact_row['start_time']
            end_time = contact_row['end_time']
            
            if start_time <= motion_time < end_time:
                contact_predictions[i] = contact_row['contact']
                contact_confidences[i] = contact_row['confidence']
                break
        else:
            # If no window found, use the last window's values
            if len(contact_df) > 0:
                last_row = contact_df.iloc[-1]
                contact_predictions[i] = last_row['contact']
                contact_confidences[i] = last_row['confidence']
    
    print(f"   📊 Aligned {len(contact_predictions)} contact predictions")
    print(f"   📊 Contact distribution: {np.bincount(contact_predictions)}")
    
    return contact_predictions, contact_confidences

def generate_primitives_simple(motion_predictions, contact_predictions, times, future_window=2.0):
    """Generate primitives using simple motion + contact logic."""
    print("🔄 Generating primitives...")
    
    primitives = []
    n = len(motion_predictions)
    
    for i in range(n):
        motion = motion_predictions[i]
        contact = contact_predictions[i]
        
        if motion and not contact:
            # Check if there's contact in the future window
            reach = False
            for j in range(i + 1, min(i + int(future_window * 30), n)):  # 30 FPS
                if contact_predictions[j]:
                    reach = True
                    break
            primitive = "reach" if reach else "reposition"
        elif motion and contact:
            primitive = "transport"
        elif not motion and contact:
            primitive = "stabilize"
        else:  # not motion and not contact
            primitive = "idle"
        
        primitives.append(primitive)
    
    print(f"   📊 Generated {len(primitives)} primitives")
    print(f"   📊 Primitive distribution: {pd.Series(primitives).value_counts().to_dict()}")
    
    return primitives

def main():
    """Main function to fix data integration and generate correct metrics."""
    print("🔧 Fixing Data Integration")
    print("=" * 60)
    
    # Load data
    contact_csv = "test_results_function_based_v2/contact_detection/S00027_feeding1_1/S00027_feeding1_1_window_contact.csv"
    motion_csv = "test_results_function_based_v2/motion_analysis/S00027_feeding1_1/S00027_feeding1_1_enhanced_motion_data.csv"
    final_csv = "test_results_function_based_v2/final_results/S00027_feeding1_1_final_predictions.csv"
    
    print(f"📁 Loading contact data: {contact_csv}")
    contact_df = pd.read_csv(contact_csv)
    print(f"   Contact data: {contact_df.shape[0]} windows")
    
    print(f"📁 Loading motion data: {motion_csv}")
    motion_df = pd.read_csv(motion_csv)
    print(f"   Motion data: {motion_df.shape[0]} frames")
    
    # Align contact data with motion data
    contact_predictions, contact_confidences = align_contact_with_motion(contact_df, motion_df)
    
    # Generate primitives
    motion_predictions = motion_df['prediction'].values
    times = motion_df['time_s'].values
    primitives = generate_primitives_simple(motion_predictions, contact_predictions, times)
    
    # Create integrated dataframe
    integrated_df = motion_df.copy()
    integrated_df['contact_prediction'] = contact_predictions
    integrated_df['contact_confidence'] = contact_confidences
    integrated_df['primitive'] = primitives
    
    # Save integrated data
    output_path = "test_results_function_based_v2/final_results/S00027_feeding1_1_fixed_predictions.csv"
    integrated_df.to_csv(output_path, index=False)
    print(f"💾 Saved integrated data to: {output_path}")
    
    # Load ground truth
    print("\n📋 Loading ground truth...")
    from utils import LabelUtils
    
    label_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/S00027/S00027_feeding1_1.csv"
    action_seq = LabelUtils.convert_labels_to_action_sequence(label_path, "left")
    gt_primitives = [action[1] for action in action_seq]
    gt_times = [action[0] for action in action_seq]
    
    print(f"   Ground truth primitives: {len(gt_primitives)}")
    print(f"   Ground truth distribution: {pd.Series(gt_primitives).value_counts().to_dict()}")
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    from primitives_utils import get_primitives_score
    
    metrics = get_primitives_score(primitives, gt_primitives)
    print(f"   Edit Score: {metrics['edit_score']:.4f}")
    print(f"   Action Error Rate: {metrics['action_error_rate']:.4f}")
    
    # Create summary
    print("\n" + "="*60)
    print("📊 CORRECTED RESULTS")
    print("="*60)
    print(f"Video ID: S00027_feeding1_1")
    print(f"Activity: feeding")
    print(f"Handedness: L")
    print(f"Motion Algorithm: windowed")
    print(f"Contact Model: Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Output Directory: test_results_function_based_v2")
    print()
    print("METRICS (CORRECTED):")
    print(f"  Edit Score: {metrics['edit_score']:.4f}")
    print(f"  Action Error Rate: {metrics['action_error_rate']:.4f}")
    print()
    print("DETAILED ANALYSIS:")
    print(f"  Total Frames: {len(integrated_df)}")
    print(f"  Predicted Actions: {len(primitives)}")
    print(f"  Ground Truth Actions: {len(gt_primitives)}")
    print(f"  Frame-level Motion Accuracy: {(motion_predictions == motion_df['ground_truth']).sum() / len(motion_predictions) * 100:.2f}%")
    print()
    print("Files Generated:")
    print(f"  Fixed CSV: {output_path}")
    print(f"  Contact CSV: {contact_csv}")
    print(f"  Motion CSV: {motion_csv}")

if __name__ == "__main__":
    main()
