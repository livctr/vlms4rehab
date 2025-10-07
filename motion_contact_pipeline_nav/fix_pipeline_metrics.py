#!/usr/bin/env python3
"""
Script to fix the pipeline metrics by recomputing them correctly
and updating the pipeline summary.

Usage:
    python fix_pipeline_metrics.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from utils import LabelUtils
from main import compute_metrics, deduplicate_sequence

def fix_metrics_for_video(video_id, final_csv_path):
    """Fix metrics for a single video and return the corrected pipeline summary."""
    print(f"\n🔧 Fixing metrics for {video_id}...")
    
    try:
        # Load the final predictions CSV
        df = pd.read_csv(final_csv_path)
        print(f"   📊 Loaded predictions: {df.shape[0]} frames")
        
        # Get predicted primitives (deduplicated)
        pred_primitives, counts = deduplicate_sequence(df)
        print(f"   🎯 Predicted primitives: {len(pred_primitives)} actions")
        
        # Get ground truth primitives
        base_id = video_id.split('_')[0]
        label_path = f"/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/{base_id}/{video_id}.csv"
        
        if not os.path.exists(label_path):
            print(f"   ❌ Ground truth file not found: {label_path}")
            return None
        
        # Determine handedness (assume left for now)
        handedness = "left"
        
        action_seq = LabelUtils.convert_labels_to_action_sequence(label_path, handedness)
        gt_primitives = [action[1] for action in action_seq]
        print(f"   📋 Ground truth primitives: {len(gt_primitives)} actions")
        
        # Compute metrics
        metrics = compute_metrics(pred_primitives, gt_primitives)
        print(f"   ✅ Corrected metrics: {metrics}")
        
        # Generate corrected pipeline summary
        video_path = f"/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/{base_id}/{video_id}.avi"
        activity = "feeding"  # This could be extracted from metadata
        handedness_display = "L"
        motion_algorithm = "windowed"
        contact_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        output_dir = "test_results_function_based"
        
        summary = f"""
============================================================
PIPELINE SUMMARY (CORRECTED)
============================================================
Video ID: {video_id}
Video: {video_path}
Activity: {activity}
Handedness: {handedness_display}
Motion Algorithm: {motion_algorithm}
Contact Model: {contact_model}
Output Directory: {output_dir}

METRICS (CORRECTED):
  Edit Score: {metrics['edit_score']:.4f}
  Action Error Rate: {metrics['action_error_rate']:.4f}

Files Generated:
  Final CSV: {output_dir}/final_results/{video_id}_final_predictions.csv

DETAILED ANALYSIS:
  Total Frames: {df.shape[0]}
  Predicted Actions: {len(pred_primitives)}
  Ground Truth Actions: {len(gt_primitives)}
  Frame-level Accuracy: {((df['prediction'] == df['ground_truth']).sum() / len(df)) * 100:.2f}%
  
  Predicted Action Sequence (first 10):
    {pred_primitives[:10]}
  
  Ground Truth Action Sequence (first 10):
    {gt_primitives[:10]}
"""
        
        return {
            'video_id': video_id,
            'corrected_metrics': metrics,
            'summary': summary,
            'pred_primitives': pred_primitives,
            'gt_primitives': gt_primitives,
            'frame_accuracy': (df['prediction'] == df['ground_truth']).sum() / len(df)
        }
        
    except Exception as e:
        print(f"   ❌ Error processing {video_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to fix metrics and generate corrected summary."""
    print("🔧 Fixing Pipeline Metrics")
    print("=" * 60)
    
    # Find video data
    final_results_dir = "test_results_function_based/final_results"
    if not os.path.exists(final_results_dir):
        print(f"❌ Final results directory not found: {final_results_dir}")
        return
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(final_results_dir) if f.endswith('_final_predictions.csv')]
    
    if not csv_files:
        print("❌ No prediction CSV files found")
        return
    
    print(f"📁 Found {len(csv_files)} videos to fix:")
    for csv_file in csv_files:
        video_id = csv_file.replace('_final_predictions.csv', '')
        print(f"   - {video_id}")
    
    # Process each video
    results = []
    for csv_file in csv_files:
        video_id = csv_file.replace('_final_predictions.csv', '')
        final_csv_path = os.path.join(final_results_dir, csv_file)
        
        result = fix_metrics_for_video(video_id, final_csv_path)
        if result:
            results.append(result)
    
    # Print corrected summaries
    print("\n" + "=" * 60)
    print("📊 CORRECTED PIPELINE SUMMARIES")
    print("=" * 60)
    
    for result in results:
        print(result['summary'])
        print("\n" + "-" * 60)
    
    # Save corrected metrics
    if results:
        corrected_df = []
        for result in results:
            corrected_df.append({
                'video_id': result['video_id'],
                'edit_score_corrected': result['corrected_metrics']['edit_score'],
                'action_error_rate_corrected': result['corrected_metrics']['action_error_rate'],
                'frame_accuracy': result['frame_accuracy'],
                'pred_actions_count': len(result['pred_primitives']),
                'gt_actions_count': len(result['gt_primitives'])
            })
        
        df = pd.DataFrame(corrected_df)
        df.to_csv("corrected_metrics.csv", index=False)
        print(f"\n💾 Corrected metrics saved to: corrected_metrics.csv")
        
        # Print overall statistics
        print(f"\n📈 CORRECTED OVERALL STATISTICS:")
        print(f"   Average Edit Score: {df['edit_score_corrected'].mean():.4f}")
        print(f"   Average Action Error Rate: {df['action_error_rate_corrected'].mean():.4f}")
        print(f"   Average Frame Accuracy: {df['frame_accuracy'].mean() * 100:.2f}%")
        print(f"   Total videos processed: {len(df)}")

if __name__ == "__main__":
    main()
