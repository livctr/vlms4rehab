#!/usr/bin/env python3
"""
Batch Processing Script for Stroke Rehabilitation Analysis

This script processes multiple videos using the main pipeline.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from main import run_complete_pipeline

def process_video_batch(metadata_csv: str, video_base_path: str, label_base_path: str, 
                       output_dir: str = "batch_results", n_videos: int = 5, **kwargs):
    """Process a batch of videos from metadata CSV."""
    
    # Load metadata
    df = pd.read_csv(metadata_csv)
    
    # Sample videos
    if n_videos > 0:
        df = df.sample(n=min(n_videos, len(df)))
    
    results_summary = []
    
    for idx, row in df.iterrows():
        video_id = row['id']
        video_path = os.path.join(video_base_path, row['path_v'])
        label_path = os.path.join(label_base_path, row['path_l']) if pd.notna(row['path_l']) else None
        activity = str(row.get('activity', 'unknown'))
        
        print(f"\n{'='*60}")
        print(f"Processing video {idx+1}/{len(df)}: {video_id}")
        print(f"Activity: {activity}")
        print(f"Video: {video_path}")
        print(f"Labels: {label_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            continue
            
        if not label_path or not os.path.exists(label_path):
            print(f"❌ Label file not found: {label_path}")
            continue
        
        try:
            # Run pipeline for this video
            video_output_dir = os.path.join(output_dir, video_id)
            results = run_complete_pipeline(
                video_path=video_path,
                label_path=label_path,
                activity=activity,
                output_dir=video_output_dir,
                **kwargs
            )
            
            # Add to summary
            summary = {
                'video_id': video_id,
                'activity': activity,
                'success': True,
                'metrics': results.get('metrics', {}),
                'output_dir': video_output_dir
            }
            results_summary.append(summary)
            
            print(f"✅ Successfully processed {video_id}")
            if 'metrics' in results:
                print(f"   Edit Score: {results['metrics'].get('edit_score', 0.0):.4f}")
                print(f"   Action Error Rate: {results['metrics'].get('action_error_rate', 0.0):.4f}")
                
        except Exception as e:
            print(f"❌ Failed to process {video_id}: {e}")
            summary = {
                'video_id': video_id,
                'activity': activity,
                'success': False,
                'error': str(e),
                'output_dir': None
            }
            results_summary.append(summary)
    
    # Save batch summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, "batch_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Compute overall statistics
    successful = summary_df[summary_df['success'] == True]
    if len(successful) > 0:
        avg_edit_score = successful['metrics'].apply(lambda x: x.get('edit_score', 0.0) if isinstance(x, dict) else 0.0).mean()
        avg_aer = successful['metrics'].apply(lambda x: x.get('action_error_rate', 0.0) if isinstance(x, dict) else 0.0).mean()
        
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total videos: {len(df)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(df) - len(successful)}")
        print(f"Average Edit Score: {avg_edit_score:.4f}")
        print(f"Average Action Error Rate: {avg_aer:.4f}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Batch Process Stroke Rehabilitation Videos")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--video_base_path", type=str, required=True, help="Base path for videos")
    parser.add_argument("--label_base_path", type=str, required=True, help="Base path for labels")
    parser.add_argument("--output_dir", type=str, default="batch_results", help="Output directory")
    parser.add_argument("--n_videos", type=int, default=5, help="Number of videos to process")
    parser.add_argument("--subsample_fps", type=int, default=15, help="Analysis FPS")
    parser.add_argument("--algo", type=str, default="windowed", choices=["hmm", "hybrid", "windowed"], help="Motion detection algorithm")
    parser.add_argument("--window_s", type=float, default=1.0, help="Window size for contact detection")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap for contact detection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model for contact detection")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")
    if not os.path.exists(args.video_base_path):
        raise FileNotFoundError(f"Video base path not found: {args.video_base_path}")
    if not os.path.exists(args.label_base_path):
        raise FileNotFoundError(f"Label base path not found: {args.label_base_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process batch
    results = process_video_batch(
        metadata_csv=args.metadata_csv,
        video_base_path=args.video_base_path,
        label_base_path=args.label_base_path,
        output_dir=args.output_dir,
        n_videos=args.n_videos,
        subsample_fps=args.subsample_fps,
        algo=args.algo,
        window_s=args.window_s,
        overlap=args.overlap,
        model=args.model
    )


if __name__ == "__main__":
    main()
