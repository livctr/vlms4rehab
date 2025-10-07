#!/usr/bin/env python3
"""
Script to generate a visualization plot comparing predicted and ground truth primitives.

Usage:
    python plot_primitives_comparison.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from utils import LabelUtils
from main import deduplicate_sequence

def create_primitives_plot(video_id, final_csv_path, output_dir="plots"):
    """Create a visualization plot comparing predicted and ground truth primitives."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Creating primitives comparison plot for {video_id}...")
    
    try:
        # Load the final predictions CSV
        df = pd.read_csv(final_csv_path)
        print(f"   📊 Loaded predictions: {df.shape[0]} frames")
        
        # Get predicted primitives (deduplicated)
        pred_primitives, pred_counts = deduplicate_sequence(df)
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
        gt_times = [action[0] for action in action_seq]
        print(f"   📋 Ground truth primitives: {len(gt_primitives)} actions")
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Primitives Comparison: {video_id}', fontsize=16, fontweight='bold')
        
        # Define colors for each primitive type
        primitive_colors = {
            'idle': '#E0E0E0',
            'reach': '#FF6B6B', 
            'transport': '#4ECDC4',
            'stabilize': '#45B7D1',
            'reposition': '#96CEB4'
        }
        
        # Plot 1: Predicted Primitives Timeline
        ax1.set_title('Predicted Primitives Timeline', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted Actions', fontsize=12)
        
        # Create timeline for predicted primitives
        pred_times = []
        pred_colors = []
        pred_labels = []
        
        current_time = 0
        for i, (primitive, count) in enumerate(zip(pred_primitives, pred_counts)):
            # Each primitive gets a time segment based on its count
            duration = count * (1/15.0)  # Assuming 15 FPS
            pred_times.append((current_time, current_time + duration))
            pred_colors.append(primitive_colors.get(primitive, '#CCCCCC'))
            pred_labels.append(primitive)
            current_time += duration
        
        # Plot predicted primitives as horizontal bars
        for i, ((start, end), color, label) in enumerate(zip(pred_times, pred_colors, pred_labels)):
            ax1.barh(0, end - start, left=start, height=0.8, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            # Add label in the middle of each bar
            mid_time = (start + end) / 2
            ax1.text(mid_time, 0, label, ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax1.set_xlim(0, max([end for start, end in pred_times]))
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_yticks([])
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Ground Truth Primitives Timeline
        ax2.set_title('Ground Truth Primitives Timeline', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Ground Truth Actions', fontsize=12)
        
        # Create timeline for ground truth primitives
        gt_colors = [primitive_colors.get(prim, '#CCCCCC') for prim in gt_primitives]
        
        # Plot ground truth primitives as horizontal bars
        for i, (time, primitive) in enumerate(zip(gt_times, gt_primitives)):
            if i < len(gt_times) - 1:
                start_time = time
                end_time = gt_times[i + 1]
            else:
                start_time = time
                end_time = time + 1.0  # Default duration for last action
            
            color = primitive_colors.get(primitive, '#CCCCCC')
            ax2.barh(0, end_time - start_time, left=start_time, height=0.8, 
                    color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            # Add label in the middle of each bar
            mid_time = (start_time + end_time) / 2
            ax2.text(mid_time, 0, primitive, ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax2.set_xlim(0, max(gt_times) + 1)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Side-by-side Comparison
        ax3.set_title('Side-by-side Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Actions', fontsize=12)
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        
        # Plot predicted primitives
        for i, ((start, end), color, label) in enumerate(zip(pred_times, pred_colors, pred_labels)):
            ax3.barh(1, end - start, left=start, height=0.4, color=color, alpha=0.7, 
                    edgecolor='black', linewidth=0.5, label=label if i == 0 else "")
        
        # Plot ground truth primitives
        for i, (time, primitive) in enumerate(zip(gt_times, gt_primitives)):
            if i < len(gt_times) - 1:
                start_time = time
                end_time = gt_times[i + 1]
            else:
                start_time = time
                end_time = time + 1.0
            
            color = primitive_colors.get(primitive, '#CCCCCC')
            ax3.barh(0, end_time - start_time, left=start_time, height=0.4, 
                    color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax3.set_xlim(0, max(max([end for start, end in pred_times]), max(gt_times) + 1))
        ax3.set_ylim(-0.5, 1.5)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Ground Truth', 'Predicted'])
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=primitive) 
                          for primitive, color in primitive_colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add statistics text
        stats_text = f"""Statistics:
Predicted Actions: {len(pred_primitives)}
Ground Truth Actions: {len(gt_primitives)}
Total Duration: {max(max([end for start, end in pred_times]), max(gt_times) + 1):.1f}s"""
        
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f"{video_id}_primitives_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Plot saved to: {output_path}")
        
        # Also save as PDF for better quality
        pdf_path = os.path.join(output_dir, f"{video_id}_primitives_comparison.pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"   💾 PDF saved to: {pdf_path}")
        
        plt.show()
        
        return output_path
        
    except Exception as e:
        print(f"   ❌ Error creating plot for {video_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_summary_plot(video_id, final_csv_path, output_dir="plots"):
    """Create a summary plot showing primitive counts and accuracy."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Creating summary plot for {video_id}...")
    
    try:
        # Load data
        df = pd.read_csv(final_csv_path)
        pred_primitives, pred_counts = deduplicate_sequence(df)
        
        base_id = video_id.split('_')[0]
        label_path = f"/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/{base_id}/{video_id}.csv"
        action_seq = LabelUtils.convert_labels_to_action_sequence(label_path, "left")
        gt_primitives = [action[1] for action in action_seq]
        
        # Count primitives
        pred_counts_dict = {}
        for prim in pred_primitives:
            pred_counts_dict[prim] = pred_counts_dict.get(prim, 0) + 1
        
        gt_counts_dict = {}
        for prim in gt_primitives:
            gt_counts_dict[prim] = gt_counts_dict.get(prim, 0) + 1
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Primitives Summary: {video_id}', fontsize=16, fontweight='bold')
        
        # Plot 1: Count comparison
        all_primitives = set(list(pred_counts_dict.keys()) + list(gt_counts_dict.keys()))
        primitives = sorted(list(all_primitives))
        
        pred_values = [pred_counts_dict.get(prim, 0) for prim in primitives]
        gt_values = [gt_counts_dict.get(prim, 0) for prim in primitives]
        
        x = np.arange(len(primitives))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pred_values, width, label='Predicted', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, gt_values, width, label='Ground Truth', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Primitive Types')
        ax1.set_ylabel('Count')
        ax1.set_title('Primitive Counts Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(primitives, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Plot 2: Accuracy by primitive type
        accuracy_by_primitive = {}
        for prim in primitives:
            pred_count = pred_counts_dict.get(prim, 0)
            gt_count = gt_counts_dict.get(prim, 0)
            if gt_count > 0:
                accuracy = min(pred_count, gt_count) / max(pred_count, gt_count)
            else:
                accuracy = 0 if pred_count == 0 else 0  # No ground truth, so 0 accuracy
            accuracy_by_primitive[prim] = accuracy
        
        accuracies = [accuracy_by_primitive[prim] for prim in primitives]
        colors = ['green' if acc > 0.5 else 'orange' if acc > 0.2 else 'red' for acc in accuracies]
        
        bars = ax2.bar(primitives, accuracies, color=colors, alpha=0.7)
        ax2.set_xlabel('Primitive Types')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy by Primitive Type')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f"{video_id}_primitives_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Summary plot saved to: {output_path}")
        
        plt.show()
        
        return output_path
        
    except Exception as e:
        print(f"   ❌ Error creating summary plot for {video_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to create plots for all videos."""
    print("🎨 Creating Primitives Comparison Plots")
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
    
    print(f"📁 Found {len(csv_files)} videos to plot:")
    for csv_file in csv_files:
        video_id = csv_file.replace('_final_predictions.csv', '')
        print(f"   - {video_id}")
    
    # Create plots for each video
    plot_paths = []
    for csv_file in csv_files:
        video_id = csv_file.replace('_final_predictions.csv', '')
        final_csv_path = os.path.join(final_results_dir, csv_file)
        
        # Create timeline comparison plot
        timeline_path = create_primitives_plot(video_id, final_csv_path)
        if timeline_path:
            plot_paths.append(timeline_path)
        
        # Create summary plot
        summary_path = create_summary_plot(video_id, final_csv_path)
        if summary_path:
            plot_paths.append(summary_path)
    
    print(f"\n✅ Created {len(plot_paths)} plots:")
    for path in plot_paths:
        print(f"   📊 {path}")

if __name__ == "__main__":
    main()
