#!/usr/bin/env python3
"""
Process existing pipeline results to generate correct metrics and plots
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from utils import LabelUtils

def load_ground_truth_primitives(label_path: str, handedness: str):
    """Load and convert ground truth labels to primitives."""
    print("📋 Loading ground truth primitives...")
    
    try:
        # Convert handedness format from 'L'/'R' to 'left'/'right'
        handedness_lower = 'left' if handedness.upper() == 'L' else 'right'
        action_seq = LabelUtils.convert_labels_to_action_sequence(label_path, handedness_lower)
        primitives = [action[1] for action in action_seq]  # Extract action names
        times = [action[0] for action in action_seq]  # Extract times
        print(f"   📊 Loaded {len(primitives)} ground truth primitives")
        return primitives, times
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
        return [], []

def compute_edit_score(pred_seq, gt_seq):
    """Compute Edit Score (ES) - percentage of correct predictions."""
    if not gt_seq:
        return 0.0
    
    # Pad sequences to same length
    max_len = max(len(pred_seq), len(gt_seq))
    pred_padded = pred_seq + ['idle'] * (max_len - len(pred_seq))
    gt_padded = gt_seq + ['idle'] * (max_len - len(gt_seq))
    
    # Count correct predictions
    correct = sum(1 for p, g in zip(pred_padded, gt_padded) if p == g)
    return (correct / max_len) * 100

def compute_action_error_rate(pred_seq, gt_seq):
    """Compute Action Error Rate (AER) - average number of errors per ground truth action."""
    if not gt_seq:
        return 0.0
    
    # Use dynamic programming for sequence alignment
    m, n = len(pred_seq), len(gt_seq)
    dp = np.zeros((m + 1, n + 1))
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_seq[i-1] == gt_seq[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # AER is total errors divided by ground truth length
    return dp[m][n] / n if n > 0 else 0.0

def create_primitives_visualization(pred_primitives, pred_times, gt_primitives, gt_times, output_path):
    """Create visualization comparing predicted and ground truth primitives."""
    print("🎨 Creating primitives visualization...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Color mapping for primitives
    primitive_colors = {
        'reach': '#FF6B6B',
        'transport': '#4ECDC4', 
        'stabilize': '#45B7D1',
        'reposition': '#96CEB4',
        'idle': '#FECA57'
    }
    
    # Plot predicted primitives
    if pred_primitives and pred_times:
        pred_colors = [primitive_colors.get(p, '#CCCCCC') for p in pred_primitives]
        ax1.bar(range(len(pred_primitives)), [1] * len(pred_primitives), 
                color=pred_colors, alpha=0.7, width=0.8)
        ax1.set_title('Predicted Primitives', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted')
        ax1.set_xlim(0, len(pred_primitives))
        ax1.set_ylim(0, 1.2)
    
    # Plot ground truth primitives
    if gt_primitives and gt_times:
        gt_colors = [primitive_colors.get(p, '#CCCCCC') for p in gt_primitives]
        ax2.bar(range(len(gt_primitives)), [1] * len(gt_primitives),
                color=gt_colors, alpha=0.7, width=0.8)
        ax2.set_title('Ground Truth Primitives', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Ground Truth')
        ax2.set_xlabel('Time Sequence')
        ax2.set_xlim(0, max(len(gt_primitives), len(pred_primitives)) if pred_primitives else len(gt_primitives))
        ax2.set_ylim(0, 1.2)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=primitive.title()) 
                      for primitive, color in primitive_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save plots
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"   💾 Plot saved to: {output_path}")
    print(f"   💾 PDF saved to: {output_path.replace('.png', '.pdf')}")

def process_existing_results():
    """Process existing pipeline results."""
    print("🔄 Processing Existing Pipeline Results")
    print("=" * 60)
    
    # Paths
    results_dir = "pipeline_results"
    final_csv = os.path.join(results_dir, "final_results", "S00027_feeding1_1_final_predictions.csv")
    label_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/S00027/S00027_feeding1_1.csv"
    handedness = "L"
    
    # Load final predictions
    print("📊 Loading final predictions...")
    if not os.path.exists(final_csv):
        print(f"❌ Final predictions CSV not found: {final_csv}")
        return
    
    df = pd.read_csv(final_csv)
    print(f"   📈 Loaded {len(df)} prediction rows")
    
    # Extract predicted primitives
    pred_primitives = df['primitive'].tolist()
    pred_times = df['time_s'].tolist()
    
    print(f"   📊 Predicted primitives: {len(pred_primitives)}")
    print(f"   📝 First 10: {pred_primitives[:10]}")
    
    # Count primitive distribution
    from collections import Counter
    pred_counts = Counter(pred_primitives)
    print(f"   📊 Predicted distribution: {dict(pred_counts)}")
    
    # Load ground truth primitives
    gt_primitives, gt_times = load_ground_truth_primitives(label_path, handedness)
    
    if not gt_primitives:
        print("❌ No ground truth primitives loaded!")
        return
    
    print(f"   📊 Ground truth primitives: {len(gt_primitives)}")
    print(f"   📝 First 10: {gt_primitives[:10]}")
    
    # Count ground truth distribution
    gt_counts = Counter(gt_primitives)
    print(f"   📊 Ground truth distribution: {dict(gt_counts)}")
    
    # Compute metrics
    print("\n📊 Computing Metrics...")
    es = compute_edit_score(pred_primitives, gt_primitives)
    aer = compute_action_error_rate(pred_primitives, gt_primitives)
    
    print(f"   ✅ Edit Score (ES): {es:.2f}%")
    print(f"   ✅ Action Error Rate (AER): {aer:.4f}")
    
    # Create visualization
    plot_path = os.path.join(results_dir, "plots", "S00027_feeding1_1_primitives_comparison_corrected.png")
    create_primitives_visualization(pred_primitives, pred_times, gt_primitives, gt_times, plot_path)
    
    # Print summary
    print("\n" + "="*60)
    print("CORRECTED METRICS SUMMARY")
    print("="*60)
    print(f"Video: S00027_feeding1_1")
    print(f"Predicted Primitives: {len(pred_primitives)}")
    print(f"Ground Truth Primitives: {len(gt_primitives)}")
    print(f"Edit Score (ES): {es:.2f}%")
    print(f"Action Error Rate (AER): {aer:.4f}")
    print(f"Plot: {plot_path}")
    print("="*60)

if __name__ == "__main__":
    process_existing_results()
