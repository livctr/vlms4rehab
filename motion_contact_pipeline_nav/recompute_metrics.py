#!/usr/bin/env python3
"""
Script to recompute Edit Score and Action Error Rate metrics
from existing pipeline results in test_results_function_based/

Usage:
    python recompute_metrics.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from utils import LabelUtils
from main import compute_metrics, deduplicate_sequence

def find_video_data(base_dir="pipeline_results"):
    """Find all video data in the test results directory."""
    video_data = {}
    
    final_results_dir = os.path.join(base_dir, "final_results")
    if not os.path.exists(final_results_dir):
        print(f"❌ Final results directory not found: {final_results_dir}")
        return video_data
    
    # Find all CSV files in final_results
    for csv_file in os.listdir(final_results_dir):
        if csv_file.endswith('_final_predictions.csv'):
            video_id = csv_file.replace('_final_predictions.csv', '')
            video_data[video_id] = {
                'final_csv': os.path.join(final_results_dir, csv_file),
                'video_id': video_id
            }
    
    return video_data

def get_ground_truth_path(video_id):
    """Get the ground truth label file path for a video ID."""
    # Extract the base ID (e.g., S00027 from S00027_feeding1_1)
    base_id = video_id.split('_')[0]
    
    # Construct the path based on the pattern we saw
    label_path = f"/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/{base_id}/{video_id}.csv"
    
    return label_path

def recompute_metrics_for_video(video_id, final_csv_path):
    """Recompute metrics for a single video."""
    print(f"\n🔄 Processing {video_id}...")
    
    try:
        # Load the final predictions CSV
        df = pd.read_csv(final_csv_path)
        print(f"   📊 Loaded predictions: {df.shape[0]} frames")
        
        # Get predicted primitives (deduplicated)
        pred_primitives, counts = deduplicate_sequence(df)
        print(f"   🎯 Predicted primitives: {len(pred_primitives)} actions")
        print(f"   📝 First 5 predicted: {pred_primitives[:5]}")
        
        # Get ground truth primitives
        label_path = get_ground_truth_path(video_id)
        print(f"   📋 Ground truth path: {label_path}")
        
        if not os.path.exists(label_path):
            print(f"   ❌ Ground truth file not found: {label_path}")
            return None
        
        # Determine handedness (assume left for now, could be improved)
        handedness = "left"  # This could be extracted from metadata
        
        action_seq = LabelUtils.convert_labels_to_action_sequence(label_path, handedness)
        gt_primitives = [action[1] for action in action_seq]
        print(f"   📋 Ground truth primitives: {len(gt_primitives)} actions")
        print(f"   📝 First 5 ground truth: {gt_primitives[:5]}")
        
        # Compute metrics
        metrics = compute_metrics(pred_primitives, gt_primitives)
        print(f"   ✅ Metrics computed: {metrics}")
        
        return {
            'video_id': video_id,
            'pred_primitives': pred_primitives,
            'gt_primitives': gt_primitives,
            'metrics': metrics,
            'pred_counts': counts
        }
        
    except Exception as e:
        print(f"   ❌ Error processing {video_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to recompute metrics for all videos."""
    print("🚀 Recomputing metrics from pipeline_results/")
    print("=" * 60)
    
    # Find all video data
    video_data = find_video_data()
    
    if not video_data:
        print("❌ No video data found in pipeline_results/")
        return
    
    print(f"📁 Found {len(video_data)} videos to process:")
    for video_id in video_data:
        print(f"   - {video_id}")
    
    # Process each video
    results = []
    for video_id, data in video_data.items():
        result = recompute_metrics_for_video(video_id, data['final_csv'])
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 METRICS SUMMARY")
    print("=" * 60)
    
    if not results:
        print("❌ No results to display")
        return
    
    for result in results:
        video_id = result['video_id']
        metrics = result['metrics']
        pred_len = len(result['pred_primitives'])
        gt_len = len(result['gt_primitives'])
        
        print(f"\n🎬 {video_id}:")
        print(f"   Predicted actions: {pred_len}")
        print(f"   Ground truth actions: {gt_len}")
        print(f"   Edit Score: {metrics['edit_score']:.4f}")
        print(f"   Action Error Rate: {metrics['action_error_rate']:.4f}")
    
    # Save results to file
    output_file = "recomputed_metrics.csv"
    results_df = []
    
    for result in results:
        results_df.append({
            'video_id': result['video_id'],
            'pred_actions_count': len(result['pred_primitives']),
            'gt_actions_count': len(result['gt_primitives']),
            'edit_score': result['metrics']['edit_score'],
            'action_error_rate': result['metrics']['action_error_rate']
        })
    
    if results_df:
        df = pd.DataFrame(results_df)
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print overall statistics
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Average Edit Score: {df['edit_score'].mean():.4f}")
        print(f"   Average Action Error Rate: {df['action_error_rate'].mean():.4f}")
        print(f"   Total videos processed: {len(df)}")

if __name__ == "__main__":
    main()
