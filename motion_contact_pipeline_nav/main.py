#!/usr/bin/env python3
"""
Main Pipeline Script for Stroke Rehabilitation Analysis

This script orchestrates the complete pipeline:
1. RTMPose analysis for motion prediction
2. VLM-based contact detection  
3. Data integration and deduplication
4. Ground truth comparison and metrics computation (AER and Edit Score)
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_rtmpose_analysis import run_rtmpose_analysis
from motion_detection_vlm import run_vlm_motion_detection
from contact_detection_vlm import run_contact_detection, run_contact_detection_framewise
from contact_detection_vlm import unload_vlm_model
from enhanced_video_generator import EnhancedMotionVideoGenerator
from utils import LabelUtils
from primitives_utils import convert_motions_and_contacts_to_prims, dedupe_list, get_primitives_score

def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("   🧹 Cleared GPU memory cache")
    except ImportError:
        pass

def get_gpu_memory_info():
    """Get GPU memory information."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   📊 Available GPUs: {gpu_count}")
            for i in range(gpu_count):
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: {total_memory:.1f}GB total, {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            return gpu_count
    except ImportError:
        pass
    return 0


def run_rtmpose_analysis_wrapper(video_path: str, label_path: str, handedness: str, 
                        activity: str, output_dir: str, subsample_fps: int = 15, 
                        algo: str = "windowed", **kwargs) -> Dict[str, Any]:
    """Run motion analysis. Supports RTMPose or VLM-based motion when algo=vlm_motion."""
    video_id = Path(video_path).stem
    video_output_dir = Path(output_dir) / "motion_analysis" / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    if algo == "vlm_motion":
        print("🔍 Running VLM-based motion analysis...")
        # Reuse activities YAML for context
        activities_yaml = kwargs.get("activities_yaml", "activities_ground_truth.yaml")
        mvideos_dir = Path(video_output_dir) / "window_videos"
        results = run_vlm_motion_detection(
            video_path=video_path,
            activities_yaml=activities_yaml,
            activity=activity,
            handedness=handedness,
            window_s=kwargs.get("motion_window_s", kwargs.get("window_s", 1.0)),
            overlap=kwargs.get("motion_overlap", kwargs.get("overlap", 0.5)),
            model=kwargs.get("vlm_motion_model", "OpenGVLab/InternVL3-38B"),
            max_frames=kwargs.get("vlm_motion_max_frames", 8),
            output_csv=str(video_output_dir / f"{video_id}_vlm_motion.csv"),
            window_videos_dir=str(mvideos_dir),
            clear_cache=kwargs.get("clear_cache", False),
            low_memory=kwargs.get("low_memory", False),
            subsample_fps=kwargs.get("subsample_fps", 10),
            annotate_wrist=True,
            playback_speed=kwargs.get("playback_speed", 1.0)
        )
        motion_df = results["motion_df"]
        return {
            "motion_df": motion_df,
            "motion_csv": results["motion_csv"],
            "output_dir": str(video_output_dir),
            "metrics": {},
            "fps": results.get("fps", 15)
        }
    else:
        print("🔍 Running RTMPose motion analysis...")
        results = run_rtmpose_analysis(
            video_path=video_path,
            label_path=label_path,
            handedness=handedness,
            output_dir=str(video_output_dir),
            subsample_fps=subsample_fps,
            algo=algo,
            window_s=kwargs.get("motion_window_s", kwargs.get("window_s", 1.0)),
            overlap=kwargs.get("motion_overlap", kwargs.get("overlap", 0.5)),
            thresh_method=kwargs.get("thresh_method", "percentile"),
            percentile=kwargs.get("percentile", 0.75),
            mad_k=kwargs.get("mad_k", 1.5),
            activity_label=activity
        )
        motion_df = pd.read_csv(results["motion_csv"])
        return {
            "motion_df": motion_df,
            "motion_csv": results["motion_csv"],
            "output_dir": str(video_output_dir),
            "metrics": results.get("metrics", {}),
            "fps": results.get("fps", 15)
        }


def run_contact_detection_wrapper(motion_csv: str, video_path: str, activity: str, 
                         handedness: str, output_dir: str, 
                         activities_yaml: str = "activities_ground_truth.yaml",
                         contact_window_s: float = 1.0, contact_overlap: float = 0.5,
                         contact_analysis_fps: float = 0.0,
                         contact_mode: str = "windowed",
                         contact_frame_fps: float = 60.0,
                         contact_gaussian_sigma: float = 1.0,
                         contact_high_threshold: float = 0.7,
                         contact_low_threshold: float = 0.3,
                         contact_batch_size: int = 8,
                         contact_median_kernel: int = 3,
                         contact_min_run_frames: int = 3,
                         contact_gap_fill_frames: int = 2,
                         playback_speed: float = 1.0,
                         model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                         model_fallback: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                         use_cpu: bool = False, low_memory: bool = False,
                         multi_gpu: bool = False, gpu_memory_fraction: float = 0.8,
                         clear_cache: bool = False) -> Dict[str, Any]:
    """Run VLM-based contact detection with OOM handling and fallback strategies."""
    print("🤖 Running VLM contact detection...")
    
    # Check GPU memory before starting
    gpu_count = get_gpu_memory_info()
    
    # Clear cache if requested
    if clear_cache:
        clear_gpu_memory()
    
    video_id = Path(video_path).stem
    contact_output_dir = Path(output_dir) / "contact_detection" / video_id
    contact_output_dir.mkdir(parents=True, exist_ok=True)
    
    contact_csv = contact_output_dir / f"{video_id}_window_contact.csv"
    window_videos_dir = contact_output_dir / "window_videos"
    
    # Try multiple models in order of preference
    models_to_try = [model]
    if model != model_fallback:
        models_to_try.append(model_fallback)
    
    # Add smaller models as fallbacks
    small_models = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "OpenGVLab/InternVL3-38B"
    ]
    for small_model in small_models:
        if small_model not in models_to_try:
            models_to_try.append(small_model)
    
    last_error = None
    
    for i, current_model in enumerate(models_to_try):
        try:
            print(f"   🔄 Trying model {i+1}/{len(models_to_try)}: {current_model}")
            
            # Set up model parameters (only supported kwargs)
            model_kwargs = {
                "max_frames": 12,
                "output_csv": str(contact_csv),
                "window_videos_dir": str(window_videos_dir)
            }
            # Add model-specific parameters
            if "InternVL3" in str(current_model):
                model_kwargs['internvl_split'] = True if multi_gpu else False
            
            # Set environment variables for memory management
            if multi_gpu and gpu_count > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(gpu_count)))
                print(f"   🔄 Using GPUs: {list(range(gpu_count))}")
            
            if gpu_memory_fraction < 1.0:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                print(f"   🔄 Limiting GPU memory usage to {gpu_memory_fraction*100:.0f}%")
            
            # Call the appropriate contact detection mode
            if str(contact_mode).lower() == "framewise":
                results = run_contact_detection_framewise(
                    motion_csv=motion_csv,
                    video_path=video_path,
                    activities_yaml=activities_yaml,
                    activity=activity,
                    handedness=handedness,
                    frame_fps=contact_frame_fps,
                    gaussian_sigma=contact_gaussian_sigma,
                    high_threshold=contact_high_threshold,
                    low_threshold=contact_low_threshold,
                    model=current_model,
                    output_csv=str(contact_csv),
                    window_videos_dir=str(window_videos_dir),
                    batch_size=contact_batch_size,
                    median_kernel=contact_median_kernel,
                    min_run_frames=contact_min_run_frames,
                    gap_fill_frames=contact_gap_fill_frames
                )
            else:
                results = run_contact_detection(
                    motion_csv=motion_csv,
                    video_path=video_path,
                    activities_yaml=activities_yaml,
                    activity=activity,
                    handedness=handedness,
                    window_s=contact_window_s,
                    overlap=contact_overlap,
                    analysis_fps=contact_analysis_fps,
                    playback_speed=playback_speed,
                    model=current_model,
                    **model_kwargs
                )
            
            # Load contact data
            contact_df = pd.read_csv(contact_csv)
            
            print(f"   ✅ Successfully used model: {current_model}")
            return {
                "contact_df": contact_df,
                "contact_csv": str(contact_csv),
                "output_dir": str(contact_output_dir),
                "total_windows": results.get("total_windows", 0),
                "model_used": current_model
            }
            
        except Exception as e:
            last_error = e
            print(f"   ❌ Model {current_model} failed: {str(e)[:100]}...")
            
            # If it's an OOM error, try next model
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"   🔄 OOM detected, trying next model...")
                continue
            else:
                # If it's not an OOM error, re-raise it
                raise e
    
    # If all models failed, create dummy contact data
    print("   ⚠️  All models failed, creating dummy contact data...")
    return create_dummy_contact_data(motion_csv, contact_output_dir, contact_csv, contact_window_s, contact_overlap)

def create_dummy_contact_data(motion_csv: str, output_dir: Path, contact_csv: Path, window_s: float, overlap: float) -> Dict[str, Any]:
    """Create dummy contact data when VLM models fail."""
    print("   🔄 Creating dummy contact data...")
    
    # Load motion data to get time range
    motion_df = pd.read_csv(motion_csv)
    total_duration = motion_df['time_s'].max()
    
    # Create windows
    windows = []
    start_time = 0.0
    window_id = 0
    
    while start_time < total_duration:
        end_time = min(start_time + window_s, total_duration)
        
        # Simple heuristic: contact = 1 if there's motion in the window
        window_motion = motion_df[
            (motion_df['time_s'] >= start_time) & 
            (motion_df['time_s'] < end_time)
        ]['prediction'].sum()
        
        # Contact if there's significant motion in the window
        contact = 1 if window_motion > window_s * 0.3 else 0  # 30% motion threshold
        
        windows.append({
            'window_id': window_id,
            'start_time': start_time,
            'end_time': end_time,
            'motion': window_motion,
            'contact': contact,
            'confidence': 0.5,  # Low confidence for dummy data
            'rationale': f"Dummy data: motion={window_motion:.1f}"
        })
        
        start_time += window_s * (1 - overlap)
        window_id += 1
    
    # Create dataframe
    contact_df = pd.DataFrame(windows)
    
    # Save contact data
    contact_df.to_csv(contact_csv, index=False)
    
    print(f"   💾 Saved dummy contact data: {len(windows)} windows")
    
    return {
        "contact_df": contact_df,
        "contact_csv": str(contact_csv),
        "output_dir": str(output_dir),
        "total_windows": len(windows),
        "model_used": "dummy_heuristic"
    }


def align_contact_with_motion(contact_df: pd.DataFrame, motion_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Align window-based contact data with frame-based motion data.

    Overlap resolution strategy:
    - Consider ALL contact windows that cover a given frame time \(start_time <= t < end_time\)
    - Compute a confidence-weighted vote using triangular weights that favor the window center
    - Predict contact=1 if the weighted probability >= 0.5; set confidence to that probability
    - If no window covers the frame, fall back to the closest window by start_time
    """
    print("🔄 Aligning contact data with motion data...")

    motion_times = motion_df['time_s'].values

    contact_predictions = np.zeros(len(motion_times), dtype=int)
    contact_confidences = np.zeros(len(motion_times), dtype=float)

    # Ensure required columns exist; if not, create sensible defaults
    cf = contact_df.copy()
    if 'confidence' not in cf.columns:
        cf['confidence'] = 1.0

    # Pre-sort for deterministic behavior
    cf_sorted = cf.sort_values('start_time').reset_index(drop=True)

    # Vector for quick access
    starts = cf_sorted['start_time'].values if len(cf_sorted) else np.array([])
    ends = cf_sorted['end_time'].values if len(cf_sorted) else np.array([])
    contacts = cf_sorted['contact'].values if len(cf_sorted) else np.array([])
    confidences = cf_sorted['confidence'].values if len(cf_sorted) else np.array([])

    for i, t in enumerate(motion_times):
        if len(cf_sorted) == 0:
            contact_predictions[i] = 0
            contact_confidences[i] = 0.0
            continue

        # Find all windows that cover this time t
        mask_cover = (starts <= t) & (t < ends)
        if mask_cover.any():
            idxs = np.where(mask_cover)[0]
            # Triangular weight around window center: 1 at center, 0 at edges
            window_lengths = np.maximum(ends[idxs] - starts[idxs], 1e-6)
            centers = (starts[idxs] + ends[idxs]) / 2.0
            distances = np.abs(t - centers)
            half_lengths = window_lengths / 2.0
            tri_weights = np.clip(1.0 - distances / half_lengths, 0.0, 1.0)

            # Confidence-weighted vote
            weights = tri_weights * confidences[idxs]
            if weights.sum() <= 1e-9:
                # Degenerate case: fall back to highest confidence
                best_idx = idxs[np.argmax(confidences[idxs])]
                pred = int(contacts[best_idx])
                conf = float(confidences[best_idx])
            else:
                prob = float(((contacts[idxs] > 0).astype(float) * weights).sum() / weights.sum())
                pred = int(prob >= 0.5)
                conf = prob
            contact_predictions[i] = pred
            contact_confidences[i] = conf
        else:
            # No covering window: use closest by start_time as heuristic
            closest_idx = int(np.argmin(np.abs(starts - t)))
            contact_predictions[i] = int(contacts[closest_idx])
            contact_confidences[i] = float(confidences[closest_idx])

    print(f"   📊 Aligned {len(contact_predictions)} contact predictions")
    print(f"   📊 Contact distribution: {np.bincount(contact_predictions)}")

    return contact_predictions, contact_confidences

def integrate_predictions(motion_df: pd.DataFrame, contact_df: pd.DataFrame) -> pd.DataFrame:
    """Integrate motion and contact predictions into a unified dataframe with proper alignment."""
    print("🔗 Integrating motion and contact predictions...")
    
    # Create a comprehensive dataframe with frame-level predictions
    integrated_df = motion_df.copy()
    
    # Add contact predictions with proper alignment
    if not contact_df.empty:
        contact_predictions, contact_confidences = align_contact_with_motion(contact_df, motion_df)
        integrated_df['contact_prediction'] = contact_predictions
        integrated_df['contact_confidence'] = contact_confidences
    else:
        integrated_df['contact_prediction'] = 0
        integrated_df['contact_confidence'] = 0.0
    
    return integrated_df


def generate_primitives_from_integrated_data(integrated_df: pd.DataFrame, fps: float = 30.0) -> Tuple[List[str], List[float]]:
    """Generate primitives from integrated motion and contact data using proper temporal windows."""
    print("🔄 Generating primitives from integrated data...")
    
    # Extract motion and contact data
    motions = integrated_df['prediction'].tolist()
    contacts = integrated_df['contact_prediction'].tolist()
    times = integrated_df['time_s'].tolist()
    
    # Add start time (0.0) to times for proper windowing
    times_with_start = [0.0] + times
    
    # Use the proper primitive conversion function
    primitives, primitive_times = convert_motions_and_contacts_to_prims(
        motions, contacts, times_with_start, future_window=2.0
    )
    
    print(f"   📊 Generated {len(primitives)} primitives from {len(motions)} frames")
    print(f"   📝 First 5 primitives: {primitives[:5]}")
    print(f"   📊 Primitive distribution: {pd.Series(primitives).value_counts().to_dict()}")
    
    return primitives, primitive_times


def generate_primitives_with_times(motion_df: pd.DataFrame, contact_df: pd.DataFrame) -> Tuple[List[str], List[float], List[float], pd.DataFrame]:
    """Generate primitives with proper start/end times by merging motion and contact data."""
    print("🔄 Generating primitives with proper temporal alignment...")
    
    # Get motion data
    motion_times = motion_df['time_s'].values
    motion_predictions = motion_df['prediction'].values
    
    # Align contact data with motion frames
    contact_predictions, contact_confidences = align_contact_with_motion(contact_df, motion_df)
    
    # Create integrated dataframe
    integrated_df = motion_df.copy()
    integrated_df['contact_prediction'] = contact_predictions
    integrated_df['contact_confidence'] = contact_confidences
    
    # Convert to primitives using the integrated data
    motions = integrated_df['prediction'].tolist()
    contacts = integrated_df['contact_prediction'].tolist()
    times = [0.0] + integrated_df['time_s'].tolist()
    
    # Convert to primitives
    primitives, _ = convert_motions_and_contacts_to_prims(motions, contacts, times)
    
    # Deduplicate primitives and track their temporal boundaries
    deduped_primitives, counts = dedupe_list(primitives)
    
    # Generate start and end times for each primitive
    primitive_start_times = []
    primitive_end_times = []
    current_frame_idx = 0
    
    for i, count in enumerate(counts):
        # Start time: beginning of this primitive segment
        start_frame_idx = current_frame_idx
        if start_frame_idx < len(times) - 1:
            start_time = times[start_frame_idx]
        else:
            start_time = times[-1] if times else 0.0
        
        # End time: beginning of next primitive segment (or end of video)
        end_frame_idx = current_frame_idx + count
        if end_frame_idx < len(times) - 1:
            end_time = times[end_frame_idx]
        else:
            end_time = times[-1] if times else start_time + 1.0
        
        primitive_start_times.append(start_time)
        primitive_end_times.append(end_time)
        current_frame_idx += count
    
    # Create primitive sequence CSV
    primitive_sequence_data = []
    for i, (primitive, start_time, end_time, count) in enumerate(zip(deduped_primitives, primitive_start_times, primitive_end_times, counts)):
        primitive_sequence_data.append({
            'primitive_id': i + 1,
            'primitive': primitive,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'frame_count': count
        })
    
    primitive_sequence_df = pd.DataFrame(primitive_sequence_data)
    
    print(f"   📊 Generated {len(deduped_primitives)} primitives with temporal boundaries")
    print(f"   📝 First 5 primitives: {deduped_primitives[:5]}")
    print(f"   ⏰ First 5 start times: {primitive_start_times[:5]}")
    print(f"   ⏰ First 5 end times: {primitive_end_times[:5]}")
    
    return deduped_primitives, primitive_start_times, primitive_end_times, primitive_sequence_df


def load_ground_truth_primitives(label_path: str, handedness: str) -> List[str]:
    """Load and convert ground truth labels to primitives."""
    print("📋 Loading ground truth primitives...")
    
    try:
        # Convert handedness format from 'L'/'R' to 'left'/'right'
        handedness_lower = 'left' if handedness.upper() == 'L' else 'right'
        action_seq = LabelUtils.convert_labels_to_action_sequence(label_path, handedness_lower)
        primitives = [action[1] for action in action_seq]  # Extract action names
        print(f"   📊 Loaded {len(primitives)} ground truth primitives")
        return primitives
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
        return []


def compute_metrics(pred_primitives: List[str], gt_primitives: List[str]) -> Dict[str, float]:
    """Compute AER and Edit Score metrics."""
    print("📊 Computing metrics...")
    
    if not gt_primitives:
        print("   ⚠️  Warning: No ground truth primitives found, returning zero metrics")
        return {"edit_score": 0.0, "action_error_rate": 0.0}
    
    if not pred_primitives:
        print("   ⚠️  Warning: No predicted primitives found, returning zero metrics")
        return {"edit_score": 0.0, "action_error_rate": 0.0}
    
    print(f"   📊 Predicted primitives: {len(pred_primitives)}")
    print(f"   📊 Ground truth primitives: {len(gt_primitives)}")
    
    # Compute metrics using the proper function
    metrics = get_primitives_score(pred_primitives, gt_primitives)
    
    print(f"   ✅ Edit Score: {metrics['edit_score']:.4f}")
    print(f"   ✅ Action Error Rate: {metrics['action_error_rate']:.4f}")
    
    return metrics


def generate_final_csv(integrated_df: pd.DataFrame, video_id: str, output_dir: str) -> str:
    """Generate the final comprehensive CSV with all predictions."""
    print("💾 Generating final CSV...")
    
    # Select relevant columns for the final output
    final_columns = [
        'frame', 'time_s', 'prediction', 'probability', 'contact_prediction', 'contact_confidence',
        'ground_truth', 'robust_velocity', 'wrist_acceleration', 'overall_confidence', 'chain_consistency'
    ]
    
    # Filter to only include columns that exist
    available_columns = [col for col in final_columns if col in integrated_df.columns]
    final_df = integrated_df[available_columns].copy()
    
    # Add primitive predictions
    motions = final_df['prediction'].tolist()
    contacts = final_df['contact_prediction'].tolist()
    times = [0.0] + final_df['time_s'].tolist()
    
    primitives, _ = convert_motions_and_contacts_to_prims(motions, contacts, times)
    final_df['primitive'] = primitives
    
    # Save final CSV
    final_dir = Path(output_dir) / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_csv_path = final_dir / f"{video_id}_final_predictions.csv"
    final_df.to_csv(final_csv_path, index=False)
    
    return str(final_csv_path)


def create_primitives_visualization(video_id: str, pred_primitives: List[str], gt_primitives: List[str], 
                                   pred_times: List[float] = None, gt_times: List[float] = None,
                                   pred_end_times: List[float] = None, output_dir: str = "pipeline_results") -> str:
    """Create visualization plots comparing predicted and ground truth primitives."""
    print("🎨 Creating primitives visualization...")
    
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Define colors for each primitive type
        primitive_colors = {
            'idle': '#E0E0E0',
            'reach': '#FF6B6B', 
            'transport': '#4ECDC4',
            'stabilize': '#45B7D1',
            'reposition': '#96CEB4'
        }
        
        # Create the main comparison plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Primitives Comparison: {video_id}', fontsize=16, fontweight='bold')
        
        # Plot 1: Predicted Primitives Timeline
        ax1.set_title('Predicted Primitives Timeline', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted Actions', fontsize=12)
        
        # Create timeline comparison for predicted primitives (action-level with times)
        pred_colors = [primitive_colors.get(prim, '#CCCCCC') for prim in pred_primitives]
        
        if pred_times and len(pred_times) > 0:
            # Plot predicted primitives on timeline with proper durations
            for i, (primitive, color, start_time) in enumerate(zip(pred_primitives, pred_colors, pred_times)):
                # Get end time for this primitive
                if pred_end_times and i < len(pred_end_times):
                    end_time = pred_end_times[i]
                else:
                    # Fallback: use next start time or estimate duration
                    if i + 1 < len(pred_times):
                        end_time = pred_times[i + 1]
                    else:
                        end_time = start_time + 1.0
                
                duration = end_time - start_time
                
                ax1.barh(0, duration, left=start_time, height=0.8, 
                        color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                # Add primitive label in the middle of each bar
                mid_time = start_time + duration / 2
                ax1.text(mid_time, 0, primitive, ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Set x-axis limit based on the maximum end time
            max_end_time = max(pred_end_times) if pred_end_times else max(pred_times)
            ax1.set_xlim(0, max_end_time)
            ax1.set_xlabel('Time (seconds)')
        else:
            # Fallback to sequence-based plotting
            for i, (primitive, color) in enumerate(zip(pred_primitives, pred_colors)):
                ax1.barh(0, 1, left=i, height=0.8, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax1.text(i + 0.5, 0, primitive, ha='center', va='center', fontsize=8, fontweight='bold')
            
            ax1.set_xlim(0, len(pred_primitives))
            ax1.set_xlabel('Action Sequence')
        
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_yticks([])
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Ground Truth Primitives Timeline
        ax2.set_title('Ground Truth Primitives Timeline', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Ground Truth Actions', fontsize=12)
        
        # Create timeline comparison for ground truth primitives
        gt_colors = [primitive_colors.get(prim, '#CCCCCC') for prim in gt_primitives]
        
        if gt_times and len(gt_times) > 0:
            # Plot ground truth primitives on timeline
            for i, (primitive, color, start_time) in enumerate(zip(gt_primitives, gt_colors, gt_times)):
                # Calculate duration for this primitive
                if i + 1 < len(gt_times):
                    duration = gt_times[i + 1] - start_time
                else:
                    # For the last primitive, use a default duration
                    duration = 1.0
                
                ax2.barh(0, duration, left=start_time, height=0.8, 
                        color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                # Add primitive label in the middle of each bar
                mid_time = start_time + duration / 2
                ax2.text(mid_time, 0, primitive, ha='center', va='center', fontsize=8, fontweight='bold')
            
            ax2.set_xlim(0, max(gt_times) if gt_times else len(gt_primitives))
            ax2.set_xlabel('Time (seconds)')
        else:
            # Fallback to sequence-based plotting
            for i, (primitive, color) in enumerate(zip(gt_primitives, gt_colors)):
                ax2.barh(0, 1, left=i, height=0.8, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax2.text(i + 0.5, 0, primitive, ha='center', va='center', fontsize=8, fontweight='bold')
            
            ax2.set_xlim(0, len(gt_primitives))
            ax2.set_xlabel('Action Sequence')
        
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Side-by-side Comparison
        ax3.set_title('Side-by-side Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Actions', fontsize=12)
        ax3.set_xlabel('Action Sequence', fontsize=12)
        
        # Plot predicted primitives (uniform width)
        for i, (primitive, color) in enumerate(zip(pred_primitives, pred_colors)):
            ax3.barh(1, 1, left=i, height=0.4, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            # Add primitive label
            ax3.text(i + 0.5, 1, primitive, ha='center', va='center', fontsize=6, fontweight='bold')
        
        # Plot ground truth primitives (uniform width)
        for i, (primitive, color) in enumerate(zip(gt_primitives, gt_colors)):
            ax3.barh(0, 1, left=i, height=0.4, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            # Add primitive label
            ax3.text(i + 0.5, 0, primitive, ha='center', va='center', fontsize=6, fontweight='bold')
        
        # Set limits based on the longer sequence
        max_len = max(len(pred_primitives), len(gt_primitives))
        ax3.set_xlim(0, max_len)
        ax3.set_ylim(-0.5, 1.5)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Ground Truth', 'Predicted'])
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=primitive) 
                          for primitive, color in primitive_colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Calculate max time for statistics
        max_time = 0
        if pred_times and len(pred_times) > 0:
            max_time = max(max_time, max(pred_times))
        if gt_times and len(gt_times) > 0:
            max_time = max(max_time, max(gt_times))
        
        # Add statistics text
        stats_text = f"""Statistics:
Predicted Actions: {len(pred_primitives)}
Ground Truth Actions: {len(gt_primitives)}
Total Duration: {max_time:.1f}s"""
        
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = plots_dir / f"{video_id}_primitives_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Plot saved to: {output_path}")
        
        # Also save as PDF for better quality
        pdf_path = plots_dir / f"{video_id}_primitives_comparison.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"   💾 PDF saved to: {pdf_path}")
        
        plt.close()  # Close the figure to free memory
        
        return str(output_path)
        
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_complete_pipeline(video_path: str, label_path: str, activity: str, 
                         output_dir: str = "pipeline_results", handedness: str = None, 
                         skip_motion: bool = False, motion_csv: str = None, **kwargs) -> Dict[str, Any]:
    """Run the complete pipeline from video to final metrics."""
    print("🚀 Starting complete stroke rehabilitation analysis pipeline...")
    
    # Auto-detect handedness if not provided
    if handedness is None:
        try:
            handedness = 'L' if LabelUtils.get_handedness(label_path).lower().startswith('l') else 'R'
            print(f"Auto-detected handedness: {handedness}")
        except:
            handedness = 'L'
            print("Using default handedness: L")
    
    video_id = Path(video_path).stem
    results = {"video_id": video_id, "activity": activity, "handedness": handedness}
    
    try:
        # Step 1: Motion analysis (can be skipped with resume)
        if skip_motion:
            print("⏭️  Skipping motion analysis (resume mode)...")
            if motion_csv and os.path.exists(motion_csv):
                motion_df = pd.read_csv(motion_csv)
                results.update({
                    "motion_df": motion_df,
                    "motion_csv": motion_csv,
                    "output_dir": str(Path(output_dir) / "motion_analysis" / video_id),
                    "fps": float(1.0 / float(pd.Series(motion_df['time_s']).diff().median())) if 'time_s' in motion_df.columns else 15
                })
            else:
                # Try default path inside output_dir
                default_motion_csv = str(Path(output_dir) / "motion_analysis" / video_id / f"{video_id}_vlm_motion.csv")
                if os.path.exists(default_motion_csv):
                    motion_df = pd.read_csv(default_motion_csv)
                    results.update({
                        "motion_df": motion_df,
                        "motion_csv": default_motion_csv,
                        "output_dir": str(Path(output_dir) / "motion_analysis" / video_id),
                        "fps": float(1.0 / float(pd.Series(motion_df['time_s']).diff().median())) if 'time_s' in motion_df.columns else 15
                    })
                else:
                    raise FileNotFoundError("skip_motion was set but no motion CSV was found. Provide --motion_csv or ensure default motion CSV exists.")
        else:
            motion_results = run_rtmpose_analysis_wrapper(
                video_path, label_path, handedness, activity, output_dir, **kwargs
            )
            results.update(motion_results)
        
        # Step 1.5: Explicitly unload any VLM and clear cache before contact
        try:
            unload_vlm_model()
        except Exception:
            pass

        # Step 2: Contact detection
        contact_results = run_contact_detection_wrapper(
            results["motion_csv"], video_path, activity, handedness, output_dir,
            contact_window_s=kwargs.get("contact_window_s", kwargs.get("window_s", 1.0)),
            contact_overlap=kwargs.get("contact_overlap", kwargs.get("overlap", 0.5)),
            contact_mode=kwargs.get("contact_mode", "windowed"),
            contact_frame_fps=kwargs.get("contact_frame_fps", 60.0),
            contact_gaussian_sigma=kwargs.get("contact_gaussian_sigma", 1.0),
            contact_high_threshold=kwargs.get("contact_high_threshold", 0.7),
            contact_low_threshold=kwargs.get("contact_low_threshold", 0.3),
            contact_analysis_fps=kwargs.get("contact_analysis_fps", 0.0),
            playback_speed=kwargs.get("playback_speed", 1.0),
            model=kwargs.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
            model_fallback=kwargs.get("model_fallback", "Qwen/Qwen2.5-VL-7B-Instruct"),
            use_cpu=kwargs.get("use_cpu", False),
            low_memory=kwargs.get("low_memory", False),
            multi_gpu=kwargs.get("multi_gpu", False),
            gpu_memory_fraction=kwargs.get("gpu_memory_fraction", 0.8),
            clear_cache=kwargs.get("clear_cache", False)
        )
        results.update(contact_results)
        
        # Step 3: Integrate predictions (ensure contact included for video overlays)
        integrated_df = integrate_predictions(
            results["motion_df"], contact_results["contact_df"]
        )
        results["integrated_df"] = integrated_df
        
        # Step 4: Generate final CSV
        final_csv = generate_final_csv(integrated_df, video_id, output_dir)
        results["final_csv"] = final_csv
        
        # Step 5: Generate primitives with proper temporal alignment
        # Load motion and contact data separately for proper alignment
        motion_csv = results["motion_csv"]
        contact_csv = results["contact_csv"]
        
        motion_df = pd.read_csv(motion_csv)
        contact_df = pd.read_csv(contact_csv)
        
        # Generate primitives with proper start/end times
        pred_primitives, pred_start_times, pred_end_times, primitive_sequence_df = generate_primitives_with_times(motion_df, contact_df)
        
        # Save primitive sequence CSV
        primitive_sequence_csv = Path(output_dir) / "final_results" / f"{video_id}_primitive_sequence.csv"
        primitive_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
        primitive_sequence_df.to_csv(primitive_sequence_csv, index=False)
        print(f"💾 Primitive sequence saved to: {primitive_sequence_csv}")
        
        results["pred_primitives"] = pred_primitives
        results["pred_start_times"] = pred_start_times
        results["pred_end_times"] = pred_end_times
        results["primitive_sequence_csv"] = str(primitive_sequence_csv)
        
        # Step 6: Load ground truth and compute metrics
        gt_primitives = load_ground_truth_primitives(label_path, handedness)
        results["gt_primitives"] = gt_primitives
        
        # Get ground truth times for visualization
        try:
            # Convert handedness format from 'L'/'R' to 'left'/'right'
            handedness_lower = 'left' if handedness.upper() == 'L' else 'right'
            action_seq = LabelUtils.convert_labels_to_action_sequence(label_path, handedness_lower)
            gt_times = [action[0] for action in action_seq]
            results["gt_times"] = gt_times
        except:
            gt_times = None
            results["gt_times"] = None
        
        # Compute metrics using action-level primitives
        metrics = compute_metrics(pred_primitives, gt_primitives)
        results["metrics"] = metrics
        
        # Step 7: Create visualization plots (if enabled)
        plot_path = None
        if kwargs.get("generate_plots", True):
            plot_path = create_primitives_visualization(
                video_id, pred_primitives, gt_primitives, pred_start_times, gt_times, pred_end_times, output_dir
            )
        results["plot_path"] = plot_path

        # Step 8: Generate ground-truth primitive sequence CSV (for summary video)
        gt_sequence_csv = None
        try:
            if gt_primitives and results.get("gt_times"):
                gt_seq_rows = []
                for i, prim in enumerate(gt_primitives):
                    start_t = results["gt_times"][i] if i < len(results["gt_times"]) else (gt_seq_rows[-1]["end_time"] if gt_seq_rows else 0.0)
                    if i + 1 < len(results["gt_times"]):
                        end_t = results["gt_times"][i + 1]
                    else:
                        # Last segment: approximate using last motion/contact end or +1s fallback
                        end_t = max(pred_end_times[-1] if pred_end_times else start_t + 1.0, start_t + 0.5)
                    gt_seq_rows.append({
                        'primitive_id': i + 1,
                        'primitive': prim,
                        'start_time': float(start_t),
                        'end_time': float(end_t),
                        'duration': float(end_t - start_t)
                    })
                gt_sequence_df = pd.DataFrame(gt_seq_rows)
                gt_sequence_csv_path = Path(output_dir) / "final_results" / f"{video_id}_gt_primitive_sequence.csv"
                gt_sequence_csv_path.parent.mkdir(parents=True, exist_ok=True)
                gt_sequence_df.to_csv(gt_sequence_csv_path, index=False)
                gt_sequence_csv = str(gt_sequence_csv_path)
                results["gt_primitive_sequence_csv"] = gt_sequence_csv
        except Exception as e:
            print(f"   ⚠️  Could not generate GT primitive sequence CSV: {e}")

        # Step 9: Generate a single summary video with timelines
        try:
            generator = EnhancedMotionVideoGenerator(handedness)
            # Use integrated data CSV so contact is available in overlays
            integrated_csv_path = Path(output_dir) / "final_results" / f"{video_id}_integrated.csv"
            integrated_df.to_csv(integrated_csv_path, index=False)
            results["integrated_csv"] = str(integrated_csv_path)

            generator.load_data(
                motion_data_path=str(integrated_csv_path),
                keypoints_path=None,
                video_path=video_path
            )
            generator.load_sequences(
                pred_sequence_csv=results.get("primitive_sequence_csv"),
                gt_sequence_csv=gt_sequence_csv
            )
            summary_dir = Path(output_dir) / "final_results"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_video_path = summary_dir / f"{video_id}_summary.mp4"
            generator.generate_summary_video(str(summary_video_path), subsample_fps=results.get("fps", 15))
            results["summary_video"] = str(summary_video_path)
        except Exception as e:
            print(f"   ⚠️  Summary video generation failed: {e}")
        
        print("✅ Pipeline completed successfully!")
        print(f"📊 Final metrics: {metrics}")
        print(f"📁 Results saved to: {output_dir}")
        if plot_path:
            print(f"🎨 Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        results["error"] = str(e)
        raise
    
    return results


def get_paths_from_id(video_id: str, base_data_path: str = None, metadata_csv: str = None):
    """
    Get video and label paths from video ID using cleaned_metadata.csv.
    
    Args:
        video_id: The video ID (e.g., 'S00027_feeding1_1')
        base_data_path: Base path for data (optional)
        metadata_csv: Path to metadata CSV (optional)
    
    Returns:
        dict: Contains video_path, label_path, activity, and other metadata
    """
    
    # Use defaults if not provided
    if base_data_path is None:
        base_data_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/"
    if metadata_csv is None:
        metadata_csv = "cleaned_metadata.csv"
    
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
    
    if not video_exists:
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not label_exists:
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    # Prepare result
    result = {
        'video_id': video_id,
        'video_path': video_path,
        'label_path': label_path,
        'activity': row['activity'],
        'patient': row['patient'],
        'stroke': row['stroke'],
        'fps': row['fps'],
        'duration_s': row['duration_s']
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Stroke Rehabilitation Analysis Pipeline")
    
    # Two modes: either provide video_id OR provide video_path + label_path + activity
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_id", type=str, help="Video ID from cleaned_metadata.csv (e.g., S00027_feeding1_1)")
    group.add_argument("--video_path", type=str, help="Path to input video (requires --label_path and --activity)")
    
    parser.add_argument("--label_path", type=str, help="Path to ground truth labels (required if using --video_path)")
    parser.add_argument("--activity", type=str, help="Activity name (required if using --video_path)")
    parser.add_argument("--handedness", type=str, choices=['L', 'R'], help="Hand to analyze (auto-detected if not provided)")
    parser.add_argument("--output_dir", type=str, default="pipeline_results", help="Output directory")
    parser.add_argument("--subsample_fps", type=int, default=60, help="Analysis FPS")
    parser.add_argument("--algo", type=str, default="windowed", choices=["hmm", "hybrid", "windowed", "llm_windowed", "vlm_windowed", "vlm_motion"], help="Motion detection algorithm")
    # Separate windowing for motion and contact
    parser.add_argument("--window_s", type=float, default=0.5, help="[DEPRECATED] Global window size (use --motion_window_s/--contact_window_s)")
    parser.add_argument("--overlap", type=float, default=0.1, help="[DEPRECATED] Global window overlap (use --motion_overlap/--contact_overlap)")
    parser.add_argument("--motion_window_s", type=float, default=None, help="Motion window size (seconds)")
    parser.add_argument("--motion_overlap", type=float, default=None, help="Motion window overlap fraction (0-1)")
    parser.add_argument("--contact_window_s", type=float, default=None, help="Contact window size (seconds)")
    parser.add_argument("--contact_overlap", type=float, default=None, help="Contact window overlap fraction (0-1)")
    parser.add_argument("--thresh_method", type=str, default="percentile", choices=["percentile", "mad"], help="Dynamic threshold method for motion detection")
    parser.add_argument("--percentile", type=float, default=0.75, help="Percentile for percentile-based thresholding (0-1)")
    parser.add_argument("--mad_k", type=float, default=1.5, help="K multiplier for MAD-based thresholding")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model for contact detection")
    parser.add_argument("--model_fallback", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Fallback VLM model if primary model fails")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage for VLM model")
    parser.add_argument("--low_memory", action="store_true", help="Enable low memory mode for VLM")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs for VLM model")
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.8, help="Fraction of GPU memory to use (0.1-1.0)")
    parser.add_argument("--clear_cache", action="store_true", help="Clear GPU cache before loading model")
    # VLM motion (optional) for algo=vlm_windowed
    parser.add_argument("--vlm_motion_model", type=str, default="OpenGVLab/InternVL3-38B", help="VLM model to use for per-window motion (e.g., Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--activities_yaml", type=str, default="activities_ground_truth.yaml", help="Activities YAML for context (used by vlm_motion)")
    parser.add_argument("--vlm_motion_device", type=str, default=None, help="Device for VLM motion model (e.g., cuda or cpu)")
    parser.add_argument("--vlm_motion_max_frames", type=int, default=1, help="Max frames per window to sample for VLM motion")
    parser.add_argument("--base_data_path", type=str, default="/gpfs/data/schambralab/quantitativeRehabilitation/__data/", 
                       help="Base data path (used with --video_id)")
    parser.add_argument("--metadata_csv", type=str, default="cleaned_metadata.csv",
                       help="Path to metadata CSV (used with --video_id)")
    # Contact framewise mode and smoothing parameters
    parser.add_argument("--contact_mode", type=str, default="windowed", choices=["windowed", "framewise"], help="Contact detection mode")
    parser.add_argument("--contact_frame_fps", type=float, default=60.0, help="Sampling FPS for framewise contact mode")
    parser.add_argument("--contact_analysis_fps", type=float, default=0.0, help="Analysis FPS for windowed contact (0=infer from motion CSV/video)")
    parser.add_argument("--contact_gaussian_sigma", type=float, default=1.0, help="Gaussian sigma for smoothing probabilities in framewise mode")
    parser.add_argument("--contact_high_threshold", type=float, default=0.7, help="Hysteresis high threshold for contact on")
    parser.add_argument("--contact_low_threshold", type=float, default=0.3, help="Hysteresis low threshold for contact off")
    parser.add_argument("--playback_speed", type=float, default=1.0, help="Virtual playback speed for VLM window inputs (e.g., 0.5=slow down)")
    parser.add_argument("--contact_batch_size", type=int, default=8, help="Batch size for framewise contact VLM")
    parser.add_argument("--contact_median_kernel", type=int, default=3, help="Median filter kernel for framewise probs")
    parser.add_argument("--contact_min_run_frames", type=int, default=3, help="Minimum frames to keep a contact run")
    parser.add_argument("--contact_gap_fill_frames", type=int, default=2, help="Fill 0 gaps up to this many frames")
    # Resume options
    parser.add_argument("--skip_motion", action="store_true", help="Skip motion analysis and resume with existing motion CSV")
    parser.add_argument("--motion_csv", type=str, default=None, help="Path to an existing motion CSV to use when skipping motion")
    parser.add_argument("--generate_plots", action="store_true", default=True, help="Generate primitives comparison plots")
    
    args = parser.parse_args()

    print("Starting main...")
    
    # Determine mode and get paths
    if args.video_id:
        # Mode 1: Use video ID
        print(f"🔍 Looking up video ID: {args.video_id}")
        try:
            path_info = get_paths_from_id(args.video_id, args.base_data_path, args.metadata_csv)
            video_path = path_info['video_path']
            label_path = path_info['label_path']
            activity = path_info['activity']
            
            print(f"✅ Found video: {path_info['activity']} (Patient: {path_info['patient']})")
            print(f"   Video: {video_path}")
            print(f"   Labels: {label_path}")
            
        except Exception as e:
            print(f"❌ Error looking up video ID: {e}")
            return
            
    else:
        # Mode 2: Use direct paths
        if not args.label_path or not args.activity:
            parser.error("--label_path and --activity are required when using --video_path")
        
        video_path = args.video_path
        label_path = args.label_path
        activity = args.activity
        
        # Validate inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
    
    # Run complete pipeline
    results = run_complete_pipeline(
        video_path=video_path,
        label_path=label_path,
        activity=activity,
        output_dir=args.output_dir,
        handedness=args.handedness,
        subsample_fps=args.subsample_fps,
        algo=args.algo,
        window_s=args.motion_window_s if args.motion_window_s is not None else args.window_s,
        overlap=args.motion_overlap if args.motion_overlap is not None else args.overlap,
        thresh_method=args.thresh_method,
        percentile=args.percentile,
        mad_k=args.mad_k,
        model=args.model,
        model_fallback=args.model_fallback,
        use_cpu=args.use_cpu,
        low_memory=args.low_memory,
        multi_gpu=args.multi_gpu,
        gpu_memory_fraction=args.gpu_memory_fraction,
        clear_cache=args.clear_cache,
        # Contact-specific params
        contact_window_s=args.contact_window_s if args.contact_window_s is not None else args.window_s,
        contact_overlap=args.contact_overlap if args.contact_overlap is not None else args.overlap,
        contact_analysis_fps=args.contact_analysis_fps,
        contact_mode=args.contact_mode,
        contact_frame_fps=args.contact_frame_fps,
        contact_gaussian_sigma=args.contact_gaussian_sigma,
        contact_high_threshold=args.contact_high_threshold,
        contact_low_threshold=args.contact_low_threshold,
        contact_batch_size=args.contact_batch_size,
        contact_median_kernel=args.contact_median_kernel,
        contact_min_run_frames=args.contact_min_run_frames,
        contact_gap_fill_frames=args.contact_gap_fill_frames,
        generate_plots=args.generate_plots,
        # Playback for VLM window inputs
        playback_speed=args.playback_speed,
        # Resume flags
        skip_motion=args.skip_motion,
        motion_csv=args.motion_csv
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    if args.video_id:
        print(f"Video ID: {args.video_id}")
    print(f"Video: {video_path}")
    print(f"Activity: {activity}")
    print(f"Handedness: {results['handedness']}")
    print(f"Motion Algorithm: {args.algo}")
    print(f"Contact Model: {args.model}")
    print(f"Output Directory: {args.output_dir}")
    print("\nMETRICS:")
    if "metrics" in results:
        print(f"  Edit Score: {results['metrics'].get('edit_score', 0.0):.4f}")
        print(f"  Action Error Rate: {results['metrics'].get('action_error_rate', 0.0):.4f}")
    print(f"\nFiles Generated:")
    print(f"  Final CSV: {results.get('final_csv', 'N/A')}")
    if results.get('summary_video'):
        print(f"  Summary Video: {results.get('summary_video')}")
    print("="*60)


if __name__ == "__main__":
    print("Starting pipeline...")
    main()
    print("Pipeline completed successfully!")
