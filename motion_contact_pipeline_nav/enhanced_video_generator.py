#!/usr/bin/env python3
"""
Enhanced Video Generator for Multi-Keypoint Motion Prediction

This script generates a video showing the robust motion prediction using
multiple keypoints with Kalman filtering, chain consistency, and quality metrics.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse
import os
from tqdm import tqdm
import json

class EnhancedMotionVideoGenerator:
    """Generate enhanced motion prediction videos with multi-keypoint visualization."""
    
    def __init__(self, handedness='L'):
        self.handedness = handedness
        
        # Default keypoint labels for display
        if handedness == 'L':
            self.keypoint_names = ['left_shoulder', 'left_elbow', 'left_wrist']
            self.keypoint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
        else:
            self.keypoint_names = ['right_shoulder', 'right_elbow', 'right_wrist']
            self.keypoint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
        
        # Colors for different elements
        self.arm_color = (255, 255, 0)
        self.motion_color = (0, 255, 255)
        self.no_motion_color = (128, 128, 128)
        self.prediction_color = (255, 0, 255)
        
        # State about optional keypoints array (if provided)
        self.use_keypoints_array = False
        self.keypoint_indices = None  # Filled if keypoints array is used
        self.kp_array_len = None
        
        # Primitive sequences (optional)
        self.pred_sequence_df = None
        self.gt_sequence_df = None
    
    def load_data(self, motion_data_path, keypoints_path=None, video_path=None):
        """Load motion data and optionally keypoints and video."""
        # Load motion data
        self.motion_df = pd.read_csv(motion_data_path)
        
        # Load keypoints if provided (optional). Otherwise, we will use CSV positions directly.
        self.keypoints_data = None
        if keypoints_path and os.path.exists(keypoints_path):
            with open(keypoints_path, 'r') as f:
                self.keypoints_data = json.load(f)
            # Inspect one frame to decide mapping (Halpe26 vs COCO17)
            try:
                first_key = next(iter(self.keypoints_data.keys()))
                arr = np.array(self.keypoints_data[first_key])
                self.kp_array_len = arr.shape[0]
                if self.kp_array_len >= 26:
                    # Halpe26 mapping
                    if self.handedness == 'L':
                        self.keypoint_indices = [5, 6, 7]  # left_shoulder, left_elbow, left_wrist
                    else:
                        self.keypoint_indices = [2, 3, 4]  # right_shoulder, right_elbow, right_wrist
                else:
                    # COCO17 mapping
                    if self.handedness == 'L':
                        self.keypoint_indices = [5, 7, 9]
                    else:
                        self.keypoint_indices = [6, 8, 10]
                self.use_keypoints_array = True
            except Exception:
                self.keypoints_data = None
                self.use_keypoints_array = False
        
        # Load video if provided
        self.video_path = video_path
        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.cap = None
            self.width = 640
            self.height = 480
            self.fps = 15
    
    def load_sequences(self, pred_sequence_csv=None, gt_sequence_csv=None):
        """Load predicted and ground-truth primitive sequences from CSVs (optional)."""
        if pred_sequence_csv and os.path.exists(pred_sequence_csv):
            try:
                self.pred_sequence_df = pd.read_csv(pred_sequence_csv)
            except Exception:
                self.pred_sequence_df = None
        if gt_sequence_csv and os.path.exists(gt_sequence_csv):
            try:
                self.gt_sequence_df = pd.read_csv(gt_sequence_csv)
            except Exception:
                self.gt_sequence_df = None
    
    def draw_enhanced_keypoints(self, frame, frame_idx):
        """Draw enhanced keypoints with quality indicators."""
        row = self.motion_df.iloc[frame_idx] if frame_idx < len(self.motion_df) else None
        if row is None:
            return frame
        
        positions = []
        confidences = []
        
        if self.use_keypoints_array and self.keypoints_data is not None and str(frame_idx) in self.keypoints_data:
            keypoints = np.array(self.keypoints_data[str(frame_idx)])
            idxs = self.keypoint_indices or []
            for kp_idx in idxs:
                if 0 <= kp_idx < len(keypoints):
                    x, y, conf = keypoints[kp_idx]
                    positions.append((int(x), int(y)))
                    confidences.append(float(conf))
                else:
                    positions.append((0, 0))
                    confidences.append(0.0)
        else:
            # Use positions from motion CSV directly
            # Prefer generic columns 'shoulder_x', 'elbow_x', 'wrist_x'
            def get_xy(prefix):
                x = row.get(f"{prefix}_x", np.nan)
                y = row.get(f"{prefix}_y", np.nan)
                return x, y
            sx, sy = get_xy('shoulder')
            ex, ey = get_xy('elbow')
            wx, wy = get_xy('wrist')
            # Fallback to handedness-specific columns if generics are missing
            if np.isnan(sx) or np.isnan(sy):
                hand_pref = 'left' if self.handedness == 'L' else 'right'
                sx, sy = get_xy(f'{hand_pref}_shoulder')
                ex, ey = get_xy(f'{hand_pref}_elbow')
                wx, wy = get_xy(f'{hand_pref}_wrist')
            conf = float(row.get('overall_confidence', 0.8))
            for x, y in [(sx, sy), (ex, ey), (wx, wy)]:
                if np.isnan(x) or np.isnan(y):
                    positions.append((0, 0))
                    confidences.append(0.0)
                else:
                    positions.append((int(x), int(y)))
                    confidences.append(conf)
        
        # Draw arm skeleton (shoulder -> elbow -> wrist)
        valid_positions = [(x, y) for (x, y), conf in zip(positions, confidences) if conf > 0.1]
        if len(valid_positions) >= 2:
            for i in range(len(valid_positions) - 1):
                pt1, pt2 = valid_positions[i], valid_positions[i + 1]
                if pt1 != (0, 0) and pt2 != (0, 0):
                    consistency = row.get('chain_consistency', 0.5)
                    color_intensity = int(255 * max(0.0, min(1.0, consistency)))
                    arm_color = (color_intensity, color_intensity, 255)
                    cv2.line(frame, pt1, pt2, arm_color, 3)
        
        # Draw keypoints with confidence-based sizing
        for i, ((x, y), conf, color) in enumerate(zip(positions, confidences, self.keypoint_colors)):
            if conf > 0.1 and (x, y) != (0, 0):
                radius = max(3, int(8 * conf))
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)
                label = self.keypoint_names[i].split('_')[1]
                cv2.putText(frame, label, (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_motion_indicators(self, frame, frame_idx):
        """Draw motion prediction indicators and quality metrics."""
        if frame_idx >= len(self.motion_df):
            return frame
        
        row = self.motion_df.iloc[frame_idx]
        
        # Motion state indicators
        ground_truth = row.get('ground_truth', 0)
        # Support both 'prediction' and legacy 'predictions'
        prediction = int(row['prediction']) if 'prediction' in row.index else int(row.get('predictions', 0))
        # Support both 'probability' and legacy 'probabilities'
        probability = float(row['probability']) if 'probability' in row.index else float(row.get('probabilities', 0.0))
        
        # Draw motion state boxes
        box_height = 30
        box_width = 120
        
        # Ground truth box
        gt_color = self.motion_color if int(ground_truth) else self.no_motion_color
        cv2.rectangle(frame, (10, 10), (10 + box_width, 10 + box_height), gt_color, -1)
        cv2.putText(frame, f"GT: {'Motion' if int(ground_truth) else 'No Motion'}", 
                   (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Prediction box
        pred_color = self.prediction_color if prediction else self.no_motion_color
        cv2.rectangle(frame, (140, 10), (140 + box_width, 10 + box_height), pred_color, -1)
        cv2.putText(frame, f"Pred: {'Motion' if prediction else 'No Motion'}", 
                   (145, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Contact status box (if available)
        contact_val = int(row.get('contact_prediction', 0))
        contact_color = (0, 165, 255) if contact_val else self.no_motion_color  # orange for contact
        cv2.rectangle(frame, (270, 10), (270 + box_width, 10 + box_height), contact_color, -1)
        cv2.putText(frame, f"Contact: {'Yes' if contact_val else 'No'}", 
                   (275, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Probability bar
        bar_width = 200
        bar_height = 15
        bar_x, bar_y = 10, 50
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Probability fill
        fill_width = int(bar_width * max(0.0, min(1.0, float(probability))))
        prob_color = (0, int(255 * float(probability)), int(255 * (1 - float(probability))))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     prob_color, -1)
        
        cv2.putText(frame, f"Probability: {float(probability):.3f}", 
                   (bar_x, bar_y + bar_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (255, 255, 255), 1)
        
        # Quality metrics
        y_offset = 90
        metrics = [
            ('Robust Vel', row.get('robust_velocity', 0)),
            ('Chain Cons', row.get('chain_consistency', 0)),
            ('Coord Index', row.get('coordination_index', 0)),
            ('Confidence', row.get('overall_confidence', 0))
        ]
        
        for i, (name, value) in enumerate(metrics):
            y_pos = y_offset + i * 20
            # Value bar
            bar_fill = int(100 * min(float(value), 1.0))
            cv2.rectangle(frame, (10, y_pos), (10 + bar_fill, y_pos + 12), 
                         (0, 255, 0), -1)
            cv2.rectangle(frame, (10 + bar_fill, y_pos), (110, y_pos + 12), 
                         (50, 50, 50), -1)
            
            cv2.putText(frame, f"{name}: {float(value):.3f}", 
                       (120, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       (255, 255, 255), 1)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_idx}", 
                   (frame.shape[1] - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {row.get('time_s', 0):.2f}s", 
                   (frame.shape[1] - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (255, 255, 255), 1)
        
        return frame
    
    def generate_video(self, output_path, subsample_fps=10):
        """Generate enhanced motion prediction video."""
        print(f"Generating enhanced motion prediction video...")
        
        # Infer analysis FPS from motion_df time_s if not provided or invalid
        analysis_fps = None
        if subsample_fps is not None and subsample_fps > 0:
            analysis_fps = float(subsample_fps)
        else:
            if 'time_s' in self.motion_df.columns and len(self.motion_df) > 1:
                dt = np.median(np.diff(self.motion_df['time_s'].values.astype(float)))
                if dt and dt > 0:
                    analysis_fps = float(1.0 / dt)
        if analysis_fps is None or analysis_fps <= 0:
            analysis_fps = 10.0
        
        # Determine original video FPS (fallback to analysis_fps if unknown)
        original_fps = float(self.fps) if self.cap else analysis_fps
        if original_fps is None or original_fps <= 0:
            original_fps = analysis_fps
        
        # Video writer setup: use analysis_fps for smooth playback matching motion sampling
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(analysis_fps), (self.width, self.height))
        
        total_motion_frames = len(self.motion_df)
        
        # Ratio for mapping motion frame index -> original video frame index
        idx_ratio = original_fps / analysis_fps if analysis_fps > 0 else 1.0
        
        # Iterate through each row of the motion data (each corresponds to a subsampled frame)
        for motion_df_idx in tqdm(range(total_motion_frames), desc="Generating video"):
            
            # Get video frame if available
            if self.cap:
                # Compute corresponding original video frame index robustly
                original_video_frame_idx = int(round(motion_df_idx * idx_ratio))
                original_video_frame_idx = max(0, min(original_video_frame_idx, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_video_frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    # Create blank frame if video frame not available
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                # Create blank frame
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Resize frame if necessary
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Draw enhanced keypoints and motion indicators using the subsampled frame index
            frame = self.draw_enhanced_keypoints(frame, motion_df_idx)
            frame = self.draw_motion_indicators(frame, motion_df_idx)
            
            # Write frame
            out.write(frame)
        
        # Cleanup
        out.release()
        if self.cap:
            self.cap.release()
        
        print(f"Enhanced motion prediction video saved to: {output_path}")
        print(f"Video details: {analysis_fps:.1f} FPS, {total_motion_frames} frames processed")

    def _draw_timeline_overlay(self, frame, current_time_s):
        """Draw timelines for motion, contact, predicted and ground-truth primitives."""
        h, w = frame.shape[:2]
        overlay_height = 140
        pad = 8
        band_height = 24
        gap = 10
        start_y = h - overlay_height
        if start_y < 0:
            start_y = 0
        
        # Semi-transparent background panel
        panel = frame.copy()
        cv2.rectangle(panel, (0, start_y), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(panel, 0.35, frame, 0.65, 0)
        
        # Time scaling
        total_duration = float(self.motion_df['time_s'].max()) if 'time_s' in self.motion_df.columns and len(self.motion_df) > 0 else max(1.0, len(self.motion_df) / max(1.0, float(self.fps or 10)))
        def t_to_x(t):
            t_clamped = max(0.0, min(float(t), total_duration))
            return int((t_clamped / total_duration) * (w - 2 * pad)) + pad
        
        # Bands: 0=Motion, 1=Contact, 2=Pred Primitives, 3=GT Primitives
        labels = ["Motion", "Contact", "Pred Prims", "GT Prims"]
        colors = [(0, 255, 255), (255, 170, 0), (255, 0, 255), (0, 200, 255)]
        
        def draw_band(y, label):
            # Band outline
            cv2.rectangle(frame, (pad, y), (w - pad, y + band_height), (200, 200, 200), 1)
            cv2.putText(frame, label, (pad + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        
        base_y = start_y + pad
        positions = [base_y + i * (band_height + gap) for i in range(4)]
        for y, label in zip(positions, labels):
            draw_band(y, label)
        
        # 1) Motion band (frame-level)
        if 'prediction' in self.motion_df.columns and 'time_s' in self.motion_df.columns:
            times = self.motion_df['time_s'].values.astype(float)
            preds = self.motion_df['prediction'].values.astype(int)
            y = positions[0]
            last_state = 0
            seg_start_t = 0.0
            for t, p in zip(times, preds):
                if p != last_state:
                    # draw segment for last_state up to t
                    x1 = t_to_x(seg_start_t)
                    x2 = t_to_x(t)
                    if last_state == 1 and x2 > x1:
                        cv2.rectangle(frame, (x1, y), (x2, y + band_height), colors[0], -1)
                    seg_start_t = t
                    last_state = p
                if t >= current_time_s:
                    break
            # Draw final partial segment until current_time_s
            x1 = t_to_x(seg_start_t)
            x2 = t_to_x(current_time_s)
            if last_state == 1 and x2 > x1:
                cv2.rectangle(frame, (x1, y), (x2, y + band_height), colors[0], -1)
        
        # 2) Contact band (frame-level aligned)
        if 'contact_prediction' in self.motion_df.columns and 'time_s' in self.motion_df.columns:
            times = self.motion_df['time_s'].values.astype(float)
            contacts = self.motion_df['contact_prediction'].values.astype(int)
            y = positions[1]
            last_state = 0
            seg_start_t = 0.0
            for t, c in zip(times, contacts):
                if c != last_state:
                    x1 = t_to_x(seg_start_t)
                    x2 = t_to_x(t)
                    if last_state == 1 and x2 > x1:
                        cv2.rectangle(frame, (x1, y), (x2, y + band_height), colors[1], -1)
                    seg_start_t = t
                    last_state = c
                if t >= current_time_s:
                    break
            x1 = t_to_x(seg_start_t)
            x2 = t_to_x(current_time_s)
            if last_state == 1 and x2 > x1:
                cv2.rectangle(frame, (x1, y), (x2, y + band_height), colors[1], -1)
        
        # Helper to draw primitive segments up to current time
        def draw_primitive_segments(seq_df, y, color):
            if seq_df is None or seq_df.empty:
                return
            for _, row in seq_df.iterrows():
                st = float(row.get('start_time', 0.0))
                et = float(row.get('end_time', st + max(0.1, float(row.get('duration', 0.1)))))
                if st > current_time_s:
                    break
                draw_et = min(et, current_time_s)
                x1 = t_to_x(st)
                x2 = t_to_x(draw_et)
                if x2 > x1:
                    cv2.rectangle(frame, (x1, y), (x2, y + band_height), color, -1)
                    # Add short label centered if space permits
                    prim = str(row.get('primitive', ''))
                    if x2 - x1 > 40 and prim:
                        mid_x = x1 + (x2 - x1) // 2
                        cv2.putText(frame, prim, (mid_x - 20, y + int(band_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # 3) Predicted primitives
        draw_primitive_segments(self.pred_sequence_df, positions[2], colors[2])
        # 4) Ground-truth primitives
        draw_primitive_segments(self.gt_sequence_df, positions[3], colors[3])
        
        # Time cursor
        x_now = t_to_x(current_time_s)
        cv2.line(frame, (x_now, start_y), (x_now, h), (255, 255, 255), 1)
        # Ticks and labels
        for frac in np.linspace(0, 1, 6):
            tx = int(frac * (w - 2 * pad)) + pad
            cv2.line(frame, (tx, h - 5), (tx, h), (220, 220, 220), 1)
            cv2.putText(frame, f"{(frac * total_duration):.0f}s", (tx - 10, h - 8 - band_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        
        return frame

    def generate_summary_video(self, output_path, subsample_fps=10):
        """Generate a summary video with timelines for motion, contact, predicted and ground-truth primitives."""
        print("Generating summary video with timelines...")
        
        # FPS inference identical to generate_video
        analysis_fps = None
        if subsample_fps is not None and subsample_fps > 0:
            analysis_fps = float(subsample_fps)
        else:
            if 'time_s' in self.motion_df.columns and len(self.motion_df) > 1:
                dt = np.median(np.diff(self.motion_df['time_s'].values.astype(float)))
                if dt and dt > 0:
                    analysis_fps = float(1.0 / dt)
        if analysis_fps is None or analysis_fps <= 0:
            analysis_fps = 10.0
        
        original_fps = float(self.fps) if self.cap else analysis_fps
        if original_fps is None or original_fps <= 0:
            original_fps = analysis_fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(analysis_fps), (self.width, self.height))
        
        total_motion_frames = len(self.motion_df)
        idx_ratio = original_fps / analysis_fps if analysis_fps > 0 else 1.0
        
        # Iterate
        for motion_df_idx in tqdm(range(total_motion_frames), desc="Generating summary video"):
            # Get base frame
            if self.cap:
                original_video_frame_idx = int(round(motion_df_idx * idx_ratio))
                original_video_frame_idx = max(0, min(original_video_frame_idx, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_video_frame_idx)
                ret, frame = self.cap.read()
                if not ret:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Standard overlays (keypoints, motion indicators)
            frame = self.draw_enhanced_keypoints(frame, motion_df_idx)
            frame = self.draw_motion_indicators(frame, motion_df_idx)
            
            # Timelines overlay based on current time
            current_time_s = float(self.motion_df.iloc[motion_df_idx].get('time_s', motion_df_idx / max(1.0, analysis_fps)))
            frame = self._draw_timeline_overlay(frame, current_time_s)
            
            out.write(frame)
        
        out.release()
        if self.cap:
            self.cap.release()
        print(f"Summary video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced motion prediction video")
    parser.add_argument("--motion_data", type=str, required=True, 
                       help="Path to enhanced motion data CSV")
    parser.add_argument("--video_path", type=str, default=None,
                       help="Path to original video (optional)")
    parser.add_argument("--keypoints_path", type=str, default=None,
                       help="Path to keypoints JSON (optional)")
    parser.add_argument("--handedness", type=str, default="L", choices=['L', 'R'],
                       help="Hand to analyze")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output video path")
    parser.add_argument("--subsample_fps", type=int, default=0,
                       help="Analysis FPS used during extraction; if 0, inferred from motion_data time_s")
    args = parser.parse_args()
    
    # Create video generator
    generator = EnhancedMotionVideoGenerator(args.handedness)
    
    # Load data
    generator.load_data(args.motion_data, args.keypoints_path, args.video_path)
    
    # Generate video
    generator.generate_video(args.output_path, args.subsample_fps)
    
    print(f"\n✅ Enhanced motion prediction video generation complete!")
    print(f"📹 Video saved to: {args.output_path}")

if __name__ == "__main__":
    main() 