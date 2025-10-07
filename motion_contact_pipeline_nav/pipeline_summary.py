#!/usr/bin/env python3
"""
Pipeline Summary Script

This script provides an overview of the complete stroke rehabilitation analysis pipeline.
"""

import os
import sys
from pathlib import Path

def print_pipeline_overview():
    """Print a comprehensive overview of the pipeline."""
    
    print("🏥 STROKE REHABILITATION ANALYSIS PIPELINE")
    print("=" * 60)
    print()
    
    print("📋 PIPELINE COMPONENTS:")
    print("-" * 30)
    print("1. RTMPose Analysis (enhanced_rtmpose_analysis.py)")
    print("   • Extracts keypoints from video frames")
    print("   • Predicts motion using multiple algorithms (HMM, Hybrid, Windowed)")
    print("   • Generates motion predictions and probabilities")
    print()
    
    print("2. Contact Detection (contact_detection_vlm.py)")
    print("   • Uses Vision-Language Models (VLM) for contact detection")
    print("   • Supports Qwen2.5-VL and InternVL3 models")
    print("   • Analyzes hand-object contact in video windows")
    print()
    
    print("3. Video Generation (enhanced_video_generator.py)")
    print("   • Creates overlay videos with predictions")
    print("   • Shows keypoints, motion states, and quality metrics")
    print("   • Generates visualization for analysis review")
    print()
    
    print("4. Main Pipeline (main.py)")
    print("   • Orchestrates the complete analysis workflow")
    print("   • Integrates motion and contact predictions")
    print("   • Generates final CSV with all predictions")
    print("   • Computes AER and Edit Score metrics")
    print()
    
    print("5. Batch Processing (batch_process.py)")
    print("   • Processes multiple videos from metadata CSV")
    print("   • Generates batch summary and statistics")
    print("   • Enables large-scale analysis")
    print()
    
    print("6. Evaluation (vic/evaluate.py)")
    print("   • Converts predictions to action primitives")
    print("   • Computes Edit Score and Action Error Rate")
    print("   • Provides deduplication and sequence analysis")
    print()


def print_data_flow():
    """Print the data flow through the pipeline."""
    
    print("🔄 DATA FLOW:")
    print("-" * 20)
    print("Input Video + Labels")
    print("        ↓")
    print("RTMPose Analysis → Motion Predictions")
    print("        ↓")
    print("VLM Contact Detection → Contact Predictions")
    print("        ↓")
    print("Data Integration → Unified Predictions")
    print("        ↓")
    print("Primitive Conversion → Action Sequences")
    print("        ↓")
    print("Deduplication → Clean Sequences")
    print("        ↓")
    print("Ground Truth Comparison → Metrics (AER, Edit Score)")
    print("        ↓")
    print("Final CSV + Overlay Video + Results JSON")
    print()


def print_output_structure():
    """Print the output directory structure."""
    
    print("📁 OUTPUT STRUCTURE:")
    print("-" * 25)
    print("pipeline_results/")
    print("├── motion_analysis/")
    print("│   └── {video_id}/")
    print("│       ├── {video_id}_enhanced_motion_data.csv")
    print("│       ├── {video_id}_metrics.json")
    print("│       └── {video_id}_enhanced_analysis.png")
    print("├── contact_detection/")
    print("│   └── {video_id}/")
    print("│       ├── {video_id}_window_contact.csv")
    print("│       └── window_videos/")
    print("├── generated_videos/")
    print("│   └── {video_id}_overlay.mp4")
    print("└── final_results/")
    print("    ├── {video_id}_final_predictions.csv")
    print("    └── {video_id}_complete_results.json")
    print()


def print_usage_examples():
    """Print usage examples."""
    
    print("💡 USAGE EXAMPLES:")
    print("-" * 20)
    print()
    
    print("Single Video Analysis:")
    print("python main.py \\")
    print("    --video_path /path/to/video.mp4 \\")
    print("    --label_path /path/to/labels.csv \\")
    print("    --activity 'face wash' \\")
    print("    --handedness L \\")
    print("    --output_dir results")
    print()
    
    print("Batch Processing:")
    print("python batch_process.py \\")
    print("    --metadata_csv cleaned_metadata.csv \\")
    print("    --video_base_path /path/to/videos/ \\")
    print("    --label_base_path /path/to/labels/ \\")
    print("    --output_dir batch_results \\")
    print("    --n_videos 10")
    print()
    
    print("Test Pipeline:")
    print("python test_pipeline.py")
    print()
    
    print("Run Examples:")
    print("python example_usage.py")
    print()


def print_metrics_explanation():
    """Print explanation of metrics."""
    
    print("📊 METRICS EXPLANATION:")
    print("-" * 25)
    print()
    
    print("Edit Score (ES):")
    print("• Measures sequence-level accuracy using edit distance")
    print("• Range: 0.0 (worst) to 1.0 (best)")
    print("• Accounts for insertions, deletions, and substitutions")
    print()
    
    print("Action Error Rate (AER):")
    print("• Measures frame-level classification accuracy")
    print("• Range: 0.0 (best) to 1.0 (worst)")
    print("• Percentage of incorrectly classified frames")
    print()
    
    print("Action Primitives:")
    print("• reach: Motion without contact (moving toward object)")
    print("• reposition: Motion without contact (not reaching)")
    print("• transport: Motion with contact (moving object)")
    print("• stabilize: No motion with contact (holding object)")
    print("• idle: No motion without contact (resting)")
    print()


def print_file_descriptions():
    """Print descriptions of all files in the pipeline."""
    
    print("📄 FILE DESCRIPTIONS:")
    print("-" * 22)
    print()
    
    files = {
        "main.py": "Main pipeline script for single video analysis",
        "batch_process.py": "Batch processing script for multiple videos",
        "enhanced_rtmpose_analysis.py": "RTMPose motion detection and analysis",
        "contact_detection_vlm.py": "VLM-based contact detection",
        "enhanced_video_generator.py": "Video generation with overlays",
        "utils.py": "Utility functions for label processing",
        "activities_ground_truth.yaml": "Activity definitions and context",
        "cleaned_metadata.csv": "Video metadata for batch processing",
        "vic/evaluate.py": "Evaluation metrics and primitive conversion",
        "example_usage.py": "Usage examples and demonstrations",
        "test_pipeline.py": "Pipeline validation and testing",
        "requirements.txt": "Python package dependencies",
        "setup.sh": "Environment setup script",
        "README.md": "Comprehensive documentation"
    }
    
    for filename, description in files.items():
        print(f"{filename:<35} - {description}")
    print()


def main():
    """Print complete pipeline summary."""
    
    print_pipeline_overview()
    print_data_flow()
    print_output_structure()
    print_usage_examples()
    print_metrics_explanation()
    print_file_descriptions()
    
    print("🚀 GETTING STARTED:")
    print("-" * 20)
    print("1. Run setup: ./setup.sh")
    print("2. Test pipeline: python test_pipeline.py")
    print("3. Run example: python example_usage.py")
    print("4. Process your data: python main.py [options]")
    print()
    print("For detailed documentation, see README.md")


if __name__ == "__main__":
    main()
