#!/usr/bin/env python3
"""
Get video and label paths from video ID using cleaned_metadata.csv

Usage:
    python get_paths.py --video_id S00027_feeding1_1
    python get_paths.py --video_id S00027_feeding1_1 --base_data_path /custom/path
"""

import os
import argparse
import pandas as pd
from pathlib import Path

# Default paths (can be overridden)
DEFAULT_BASE_DATA_PATH = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/"
DEFAULT_METADATA_CSV = "cleaned_metadata.csv"

def get_paths_from_id(video_id: str, base_data_path: str = None, metadata_csv: str = None):
    """
    Get video and label paths from video ID.
    
    Args:
        video_id: The video ID (e.g., 'S00027_feeding1_1')
        base_data_path: Base path for data (optional)
        metadata_csv: Path to metadata CSV (optional)
    
    Returns:
        dict: Contains video_path, label_path, activity, and other metadata
    """
    
    # Use defaults if not provided
    if base_data_path is None:
        base_data_path = DEFAULT_BASE_DATA_PATH
    if metadata_csv is None:
        metadata_csv = DEFAULT_METADATA_CSV
    
    # Check if metadata CSV exists
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    
    # Load metadata
    df = pd.read_csv(metadata_csv)
    
    # Find the video ID
    video_row = df[df['id'] == video_id]
    
    if video_row.empty:
        available_ids = df['id'].head(10).tolist()
        raise ValueError(f"Video ID '{video_id}' not found in metadata. Available IDs (first 10): {available_ids}")
    
    # Get the row data
    row = video_row.iloc[0]
    
    # Build paths
    video_path = os.path.join(base_data_path, "VideoData", "rawVideosADLsandFM", row['path_v'])
    label_path = os.path.join(base_data_path, "rawVideoLabels", row['path_l']) if pd.notna(row['path_l']) else None
    
    # Check if files exist
    video_exists = os.path.exists(video_path)
    label_exists = os.path.exists(label_path) if label_path else False
    
    # Prepare result
    result = {
        'video_id': video_id,
        'video_path': video_path,
        'label_path': label_path,
        'activity': row['activity'],
        'patient': row['patient'],
        'stroke': row['stroke'],
        'fps': row['fps'],
        'duration_s': row['duration_s'],
        'video_exists': video_exists,
        'label_exists': label_exists,
        'base_data_path': base_data_path
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Get video and label paths from video ID")
    parser.add_argument("--video_id", type=str, required=True, help="Video ID (e.g., S00027_feeding1_1)")
    parser.add_argument("--base_data_path", type=str, default=DEFAULT_BASE_DATA_PATH, 
                       help=f"Base data path (default: {DEFAULT_BASE_DATA_PATH})")
    parser.add_argument("--metadata_csv", type=str, default=DEFAULT_METADATA_CSV,
                       help=f"Path to metadata CSV (default: {DEFAULT_METADATA_CSV})")
    parser.add_argument("--check_files", action="store_true", 
                       help="Check if files actually exist on disk")
    
    args = parser.parse_args()
    
    try:
        # Get paths
        result = get_paths_from_id(args.video_id, args.base_data_path, args.metadata_csv)
        
        # Print results
        print(f"Video ID: {result['video_id']}")
        print(f"Activity: {result['activity']}")
        print(f"Patient: {result['patient']}")
        print(f"Stroke: {result['stroke']}")
        print(f"FPS: {result['fps']}")
        print(f"Duration: {result['duration_s']}s")
        print()
        print(f"Video Path: {result['video_path']}")
        print(f"Label Path: {result['label_path']}")
        
        if args.check_files:
            print()
            print("File Status:")
            print(f"  Video exists: {'✅' if result['video_exists'] else '❌'}")
            print(f"  Label exists: {'✅' if result['label_exists'] else '❌'}")
            
            if not result['video_exists']:
                print(f"  ⚠️  Video file not found: {result['video_path']}")
            if not result['label_exists']:
                print(f"  ⚠️  Label file not found: {result['label_path']}")
        
        # Return the result for programmatic use
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    main()
