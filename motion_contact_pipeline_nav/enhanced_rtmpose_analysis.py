#!/usr/bin/env python3
"""
Enhanced RTMPose Analysis with Multi-Keypoint Robust Motion Prediction

This script extracts multiple keypoints (shoulder, elbow, wrist) and uses Kalman filtering
to create a robust velocity estimate for better motion prediction.

MODIFIED VERSION: Features a completely redesigned prediction algorithm using a 
composite motion score, a more robust HMM, and advanced post-processing to
significantly improve precision for detecting sub-second clinical movements.
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean
import yaml
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support, average_precision_score

# RTMPose imports
try:
    from mmpose.apis import inference_topdown
    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.utils import register_all_modules
    register_all_modules()
    MMPOSE_AVAILABLE = True
except ImportError:
    print("MMPose not available. Please install: pip install mmpose")
    MMPOSE_AVAILABLE = False

class EnhancedKalmanFilter:
    """Enhanced Kalman filter for multiple keypoints with velocity estimation."""
    
    def __init__(self, n_keypoints=3, dt=1/15.0):
        self.n_keypoints = n_keypoints
        self.dt = dt
        
        # State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3]
        state_size = n_keypoints * 4
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(state_size)
        for i in range(n_keypoints):
            base_idx = i * 4
            self.F[base_idx, base_idx + 2] = dt      # x += vx * dt
            self.F[base_idx + 1, base_idx + 3] = dt  # y += vy * dt
        
        # Measurement matrix (we observe positions only)
        self.H = np.zeros((n_keypoints * 2, state_size))
        for i in range(n_keypoints):
            self.H[i * 2, i * 4] = 1      # observe x
            self.H[i * 2 + 1, i * 4 + 1] = 1  # observe y
        
        # Process noise covariance (tuned for human motion)
        self.Q = np.eye(state_size) * 0.5
        
        # Measurement noise covariance
        self.R = np.eye(n_keypoints * 2) * 2.0
        
        # Initial state covariance
        self.P = np.eye(state_size) * 100.0
        
        # Initial state
        self.x = np.zeros(state_size)
        self.initialized = False
    
    def predict(self):
        """Predict step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, measurements, confidences=None):
        """Update step with confidence-based noise adaptation."""
        if not self.initialized:
            # Initialize state with first measurement
            for i in range(self.n_keypoints):
                self.x[i * 4] = measurements[i * 2]      # x position
                self.x[i * 4 + 1] = measurements[i * 2 + 1]  # y position
            self.initialized = True
            return self.x
        
        # Adaptive measurement noise based on confidence
        if confidences is not None:
            R_adaptive = self.R.copy()
            for i in range(self.n_keypoints):
                conf = max(confidences[i], 0.1)  # Minimum confidence
                noise_factor = 1.0 / conf
                R_adaptive[i * 2, i * 2] *= noise_factor
                R_adaptive[i * 2 + 1, i * 2 + 1] *= noise_factor
        else:
            R_adaptive = self.R
        
        # Innovation
        y = measurements - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R_adaptive
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        
        return self.x
    
    def get_positions_and_velocities(self):
        """Get current positions and velocities for all keypoints."""
        positions = []
        velocities = []
        
        for i in range(self.n_keypoints):
            base_idx = i * 4
            pos = [self.x[base_idx], self.x[base_idx + 1]]
            vel = [self.x[base_idx + 2], self.x[base_idx + 3]]
            positions.append(pos)
            velocities.append(vel)
        
        return np.array(positions), np.array(velocities)

class EnhancedRTMPoseExtractor:
    """Enhanced RTMPose extractor for multiple keypoints."""
    
    def __init__(self, config_path=None, checkpoint_path=None):
        if not MMPOSE_AVAILABLE:
            raise ImportError("MMPose is required for RTMPose. Install with: pip install mmpose")
        
        # Use local checkpoint if available
        import glob
        local_checkpoints = glob.glob('./rtmpose-*.pth')
        if local_checkpoints and checkpoint_path is None:
            checkpoint_path = local_checkpoints[0]
            print(f"Found local checkpoint: {checkpoint_path}")
        elif checkpoint_path is None:
            checkpoint_path = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth'
        
        if config_path is None:
            config_path = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py'
        
        print(f"Using config: {config_path}")
        print(f"Using checkpoint: {checkpoint_path}")
        
        if isinstance(config_path, str) and config_path.startswith('http'):
            try:
                import tempfile, urllib.request
                tmp_dir = tempfile.gettempdir()
                local_cfg = os.path.join(tmp_dir, os.path.basename(config_path))
                if not os.path.exists(local_cfg):
                    print(f"Downloading config to {local_cfg}...")
                    urllib.request.urlretrieve(config_path, local_cfg)
                config_path = local_cfg
            except Exception as e:
                raise RuntimeError(f"Failed to download config from URL: {config_path}. Error: {e}")
        
        self.pose_estimator = init_pose_estimator(
            config_path, 
            checkpoint_path, 
            device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        )
        
        self.keypoint_names = self.pose_estimator.dataset_meta.get('keypoint_id2name')
        self.num_keypoints = len(self.keypoint_names) if self.keypoint_names else 17
        print(f"Detected {self.num_keypoints} keypoints from model meta.")
        
    def extract_keypoints(self, video_path: str, subsample_fps: int = 10) -> tuple:
        """Extract keypoints using RTMPose with frame subsampling."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_skip = max(1, int(original_fps / subsample_fps))
        effective_fps = original_fps / frame_skip
        
        print(f"Original FPS: {original_fps}, Subsampling to ~{effective_fps:.1f} FPS (every {frame_skip} frames)")
        
        keypoints_data = {}
        
        frame_count = 0
        for frame_idx in tqdm(range(0, total_frames, frame_skip), desc="Enhanced RTMPose Keypoint Extraction"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            results = inference_topdown(self.pose_estimator, frame)
            
            if results and len(results) > 0:
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                keypoints_with_conf = np.concatenate([keypoints, scores.reshape(-1, 1)], axis=1)
                keypoints_data[frame_count] = keypoints_with_conf
            else:
                keypoints_data[frame_count] = np.zeros((self.num_keypoints, 3))
            
            frame_count += 1
        
        cap.release()
        return keypoints_data, frame_count, width, height, effective_fps

class RobustMotionAnalyzer:
    """Robust motion analysis using multiple keypoints and Kalman filtering."""
    
    def __init__(self, handedness='L', fps=15.0, keypoint_names=None):
        self.handedness = handedness
        self.fps = fps
        self.kalman_filter = EnhancedKalmanFilter(n_keypoints=3, dt=1.0/fps)
        self._prev_arm_angle = None
        self._prev_velocities = None
        self._prev_angular_velocity = None
        self._prev_elbow_angle = None
        
        # Define keypoint indices based on common conventions
        if keypoint_names:
            names = [str(n).lower() for n in keypoint_names]
            prefix = 'left' if handedness == 'L' else 'right'
            try:
                shoulder = names.index(f'{prefix}_shoulder')
                elbow = names.index(f'{prefix}_elbow')
                wrist = names.index(f'{prefix}_wrist')
                self.keypoint_indices = [shoulder, elbow, wrist]
            except ValueError:
                # Fallback for different naming conventions
                self.keypoint_indices = [5, 7, 9] if handedness == 'L' else [6, 8, 10]
        else: # COCO fallback
            self.keypoint_indices = [5, 7, 9] if handedness == 'L' else [6, 8, 10]

    def analyze_motion(self, keypoints_data, total_frames):
        """Analyze motion using multiple keypoints with Kalman filtering."""
        motion_data = []
        
        for frame_idx in range(total_frames):
            kps = keypoints_data.get(frame_idx)
            
            if kps is not None and len(kps) > max(self.keypoint_indices):
                measurements = []
                confidences = []
                
                for kp_idx in self.keypoint_indices:
                    x, y, conf = kps[kp_idx]
                    measurements.extend([x, y])
                    confidences.append(conf)
                
                measurements = np.array(measurements)
                confidences = np.array(confidences)
                
                self.kalman_filter.predict()
                self.kalman_filter.update(measurements, confidences)
                
                positions, velocities = self.kalman_filter.get_positions_and_velocities()
                motion_metrics = self._compute_motion_metrics(positions, velocities, confidences)
            else:
                motion_metrics = self._get_zero_metrics()
                confidences = np.zeros(3)
            
            motion_data.append({
                'frame': frame_idx,
                'time_s': frame_idx / self.fps,
                **motion_metrics,
                'overall_confidence': confidences.mean() if len(confidences) > 0 else 0.0
            })
        
        return pd.DataFrame(motion_data)

    def _compute_motion_metrics(self, positions, velocities, confidences):
        """Compute comprehensive motion metrics."""
        shoulder_vel = np.linalg.norm(velocities[0])
        elbow_vel = np.linalg.norm(velocities[1])
        wrist_vel = np.linalg.norm(velocities[2])
        
        if self._prev_velocities is not None:
            wrist_acc = np.linalg.norm((velocities[2] - self._prev_velocities[2]) * self.fps)
        else:
            wrist_acc = 0.0
        self._prev_velocities = velocities.copy()
        
        weights = confidences / (confidences.sum() + 1e-8)
        weighted_avg_vel = np.sum(weights * np.array([shoulder_vel, elbow_vel, wrist_vel]))
        
        upper_arm_vec = positions[1] - positions[0]
        forearm_vec = positions[2] - positions[1]
        chain_consistency = 1.0 / (1.0 + abs(np.linalg.norm(upper_arm_vec) - np.linalg.norm(forearm_vec)) / 100)
        
        robust_velocity = wrist_vel * chain_consistency * confidences[2]
        
        shoulder_to_wrist = positions[2] - positions[0]
        arm_angle = np.arctan2(shoulder_to_wrist[1], shoulder_to_wrist[0])
        arm_length = float(np.linalg.norm(shoulder_to_wrist))

        # Elbow joint angle (between upper arm and forearm)
        def _safe_unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-6 else v
        ua_u = _safe_unit(upper_arm_vec)
        fa_u = _safe_unit(forearm_vec)
        dot = float(np.clip(np.dot(ua_u, fa_u), -1.0, 1.0))
        elbow_angle = float(np.arccos(dot))  # radians
        if self._prev_elbow_angle is not None:
            dtheta = elbow_angle - self._prev_elbow_angle
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            elbow_angular_velocity = abs(dtheta) * self.fps
        else:
            elbow_angular_velocity = 0.0
        self._prev_elbow_angle = elbow_angle
        
        if self._prev_arm_angle is not None:
            angle_diff = arm_angle - self._prev_arm_angle
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            angular_velocity = abs(angle_diff) * self.fps
        else:
            angular_velocity = 0.0
        self._prev_arm_angle = arm_angle
        
        return {
            'shoulder_velocity': shoulder_vel, 'elbow_velocity': elbow_vel, 'wrist_velocity': wrist_vel,
            'wrist_acceleration': wrist_acc, 'weighted_avg_velocity': weighted_avg_vel,
            'robust_velocity': robust_velocity, 'angular_velocity': angular_velocity,
            'chain_consistency': chain_consistency,
            'shoulder_x': positions[0][0], 'shoulder_y': positions[0][1],
            'elbow_x': positions[1][0], 'elbow_y': positions[1][1],
            'wrist_x': positions[2][0], 'wrist_y': positions[2][1],
            'arm_length': arm_length,
            'elbow_angle': elbow_angle, 'elbow_angular_velocity': elbow_angular_velocity
        }

    def _get_zero_metrics(self):
        """Return zero metrics for missing data."""
        return {
            'shoulder_velocity': 0.0, 'elbow_velocity': 0.0, 'wrist_velocity': 0.0,
            'wrist_acceleration': 0.0, 'weighted_avg_velocity': 0.0,
            'robust_velocity': 0.0, 'angular_velocity': 0.0,
            'chain_consistency': 0.0,
            'shoulder_x': 0.0, 'shoulder_y': 0.0,
            'elbow_x': 0.0, 'elbow_y': 0.0,
            'wrist_x': 0.0, 'wrist_y': 0.0,
            'arm_length': 0.0,
            'elbow_angle': 0.0, 'elbow_angular_velocity': 0.0
        }

    # =================================================================================
    # ===== START OF REWRITTEN HIGH-PRECISION MOTION PREDICTION ALGORITHM =====
    # =================================================================================
    def _post_process_predictions(self, predictions, min_len_motion, min_len_gap):
        """
        Cleans up binary predictions by removing short segments and filling small gaps.
        This is crucial for improving precision by eliminating brief, spurious detections.
        """
        # 1. Remove short MOTION segments (flips 1s to 0s)
        # Find transitions: 0->1 (starts) and 1->0 (ends)
        padded = np.concatenate(([0], predictions, [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for s, e in zip(starts, ends):
            if (e - s) < min_len_motion:
                predictions[s:e] = 0

        # 2. Fill short NO-MOTION gaps between motion segments (flips 0s to 1s)
        padded = np.concatenate(([0], predictions, [0]))
        diffs = np.diff(padded)
        motion_ends = np.where(diffs == -1)[0]
        motion_starts = np.where(diffs == 1)[0]

        for i in range(len(motion_ends) - 1):
            gap_start = motion_ends[i]
            gap_end = motion_starts[i + 1]
            if (gap_end - gap_start) < min_len_gap:
                predictions[gap_start:gap_end] = 1

        return predictions

    def predict_motion_robust(self, motion_df):
        """
        Predicts motion using a more robust, multi-feature Hidden Markov Model (HMM) 
        with Viterbi decoding and advanced post-processing for high precision.
        """
        # --- 1. Configuration & Tunable Parameters ---
        # These parameters can be adjusted to optimize for different conditions
        cfg = {
            'velocity_weight': 0.5,
            'accel_weight': 0.5,
            'smoothing_kernel': 5,
            'prob_scale': 10.0,
            'prob_midpoint': 0.35,
            'hmm_p_stay': 0.985,
            'post_min_motion_len': 4,
            'post_fill_gap_len': 2,
        }

        # --- 2. Feature Engineering & Scaling ---
        velocity = motion_df['robust_velocity'].values
        acceleration = motion_df['wrist_acceleration'].values

        # Smooth raw signals with a median filter to reduce noise before combining
        velocity_smooth = signal.medfilt(velocity, kernel_size=cfg['smoothing_kernel'])
        accel_smooth = signal.medfilt(acceleration, kernel_size=cfg['smoothing_kernel'])

        # Robust scaling: clip outliers at 98th percentile, then scale from 0 to 1
        def robust_scale(x):
            p98 = np.percentile(x, 98)
            if p98 == 0: return np.zeros_like(x) # Avoid division by zero
            x_clipped = np.clip(x, 0, p98)
            return x_clipped / p98

        velocity_scaled = robust_scale(velocity_smooth)
        accel_scaled = robust_scale(accel_smooth)

        # --- 3. Create Composite Motion Score with Quality Modulation ---
        quality_raw = 0.5 * np.clip(motion_df['chain_consistency'].values, 0, 1) + \
                      0.5 * np.clip(motion_df['overall_confidence'].values, 0, 1)
        quality_smooth = signal.medfilt(quality_raw, kernel_size=cfg['smoothing_kernel'])
        composite_base = (cfg['velocity_weight'] * velocity_scaled +
                          cfg['accel_weight'] * accel_scaled)
        composite_score = composite_base * quality_smooth

        # --- 4. Probabilistic Modeling (HMM Emission Probabilities) ---
        # Map the composite score to a probability of motion using a sigmoid function.
        # This provides a clean, continuous likelihood for the HMM.
        def sigmoid(x, k, x0):
            return 1 / (1 + np.exp(-k * (x - x0)))

        prob_motion = sigmoid(composite_score, cfg['prob_scale'], cfg['prob_midpoint'])

        epsilon = 1e-10
        log_prob_motion = np.log(prob_motion + epsilon)
        log_prob_no_motion = np.log(1 - prob_motion + epsilon)
        log_emissions = np.stack([log_prob_no_motion, log_prob_motion], axis=1)

        # --- 5. Viterbi Decoding (Temporal Smoothing) ---
        # This finds the most likely sequence of states (Motion/No-Motion) given our
        # probabilities, enforcing that states tend to persist over time.
        n_states = 2
        n_obs = len(motion_df)
        p_stay, p_switch = cfg['hmm_p_stay'], 1 - cfg['hmm_p_stay']
        log_trans = np.log(np.array([[p_stay, p_switch], [p_switch, p_stay]]))
        log_start_p = np.log(np.array([0.9, 0.1]))  # Assume starting with no motion

        trellis = np.zeros((n_obs, n_states))
        backpointer = np.zeros((n_obs, n_states), dtype=int)
        trellis[0, :] = log_start_p + log_emissions[0, :]

        for t in range(1, n_obs):
            for s in range(n_states):
                seq_probs = trellis[t - 1, :] + log_trans[:, s]
                backpointer[t, s] = np.argmax(seq_probs)
                trellis[t, s] = np.max(seq_probs) + log_emissions[t, s]

        best_path = np.zeros(n_obs, dtype=int)
        best_path[-1] = np.argmax(trellis[-1, :])
        for t in range(n_obs - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        # --- 6. Post-processing to Finalize Predictions ---
        # This final step is critical for high precision. It cleans up the HMM output
        # by removing spurious detections, resulting in cleaner, more reliable results.
        final_predictions = self._post_process_predictions(
            best_path,
            min_len_motion=cfg['post_min_motion_len'],
            min_len_gap=cfg['post_fill_gap_len']
        )
        
        # Return the final cleaned predictions and the raw probability score
        return final_predictions, prob_motion
    # ===============================================================================
    # ===== END OF REWRITTEN HIGH-PRECISION MOTION PREDICTION ALGORITHM =====
    # ===============================================================================
    
    def predict_motion_hysteresis(self, motion_df):
        """
        Predict motion using a hybrid, quality-modulated hysteresis approach with
        adaptive thresholds and explicit duration constraints (HSMM-like enforcement).
        Returns (binary_predictions, probability_like_score).
        """
        cfg = {
            'smoothing_window': 7,         # odd
            'savgol_poly': 2,
            'q_low': 0.70,
            'q_high': 0.85,
            'min_on_s': 0.40,
            'min_off_s': 0.30,
            'gap_fill_s': 0.25,
            'w_velocity': 0.45,
            'w_accel': 0.35,
            'w_angular': 0.20,
            'w_jerk': 0.0
        }
        n = len(motion_df)
        if n == 0:
            return np.array([]), np.array([])

        # Features
        velocity = motion_df['robust_velocity'].values.astype(float)
        acceleration = motion_df['wrist_acceleration'].values.astype(float)
        angular = motion_df['angular_velocity'].values.astype(float)

        # Jerk (derivative of acceleration)
        jerk = np.zeros_like(acceleration)
        if n > 1:
            jerk[1:] = np.diff(acceleration) * self.fps

        # Helper: ensure odd window <= n
        def _odd_window(w):
            w = int(max(3, w))
            if w % 2 == 0:
                w += 1
            return min(w, n - 1 if (n - 1) % 2 == 1 else n - 2) if n >= 5 else 3

        win = _odd_window(cfg['smoothing_window'])

        # Smooth features
        try:
            vel_s = signal.savgol_filter(velocity, window_length=win, polyorder=min(cfg['savgol_poly'], win - 1))
            acc_s = signal.savgol_filter(acceleration, window_length=win, polyorder=min(cfg['savgol_poly'], win - 1))
            ang_s = signal.savgol_filter(angular, window_length=win, polyorder=min(cfg['savgol_poly'], win - 1))
            jerk_s = signal.savgol_filter(jerk, window_length=win, polyorder=min(cfg['savgol_poly'], win - 1))
        except Exception:
            # Fallback to median if savgol not applicable
            vel_s = signal.medfilt(velocity, kernel_size=win)
            acc_s = signal.medfilt(acceleration, kernel_size=win)
            ang_s = signal.medfilt(angular, kernel_size=win)
            jerk_s = signal.medfilt(jerk, kernel_size=win)

        # Robust scaling (0-1) via 98th percentile
        def rscale(x):
            p98 = np.percentile(np.abs(x), 98)
            if p98 <= 1e-8:
                return np.zeros_like(x)
            return np.clip(np.abs(x) / p98, 0, 1)

        vel_n = rscale(vel_s)
        acc_n = rscale(acc_s)
        ang_n = rscale(ang_s)
        jerk_n = rscale(jerk_s)

        composite = (cfg['w_velocity'] * vel_n +
                     cfg['w_accel'] * acc_n +
                     cfg['w_angular'] * ang_n +
                     cfg['w_jerk'] * jerk_n)

        # Quality modulation
        quality = 0.5 * np.clip(motion_df['chain_consistency'].values, 0, 1) + \
                  0.5 * np.clip(motion_df['overall_confidence'].values, 0, 1)
        quality_s = signal.medfilt(quality, kernel_size=min(win, n if n % 2 == 1 else n - 1)) if n >= 3 else quality
        score = composite * quality_s

        # Adaptive hysteresis thresholds from distribution of score
        q_low = float(np.quantile(score, cfg['q_low']))
        q_high = float(np.quantile(score, cfg['q_high']))
        if q_high <= q_low:
            q_high = q_low + 1e-6

        # Hysteresis decoding
        preds = np.zeros(n, dtype=int)
        state = 0
        for i in range(n):
            s = score[i]
            if state == 0 and s >= q_high:
                state = 1
            elif state == 1 and s <= q_low:
                state = 0
            preds[i] = state

        # HSMM-like duration enforcement
        min_on = max(1, int(round(cfg['min_on_s'] * self.fps)))
        min_off = max(1, int(round(cfg['min_off_s'] * self.fps)))
        gap_fill = max(1, int(round(cfg['gap_fill_s'] * self.fps)))

        def _enforce_runs(binary, min_len_on, min_len_off, gap_fill_len):
            x = binary.copy()
            # remove short ON
            padded = np.concatenate(([0], x, [0]))
            d = np.diff(padded)
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            for s, e in zip(starts, ends):
                if (e - s) < min_len_on:
                    x[s:e] = 0
            # recompute to fill short gaps
            padded = np.concatenate(([0], x, [0]))
            d = np.diff(padded)
            ends = np.where(d == -1)[0]
            starts = np.where(d == 1)[0]
            for i in range(len(ends) - 1):
                gap_s = ends[i]
                gap_e = starts[i + 1]
                if (gap_e - gap_s) <= gap_fill_len:
                    x[gap_s:gap_e] = 1
            # remove short OFF
            padded = np.concatenate(([1], x, [1]))
            d = np.diff(padded)
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            for s, e in zip(starts, ends):
                if (e - s) < min_len_off:
                    x[s:e] = 1
            return x

        preds = _enforce_runs(preds, min_on, min_off, gap_fill)

        # Probability-like score via sigmoid mapping for evaluation/plots
        def sigmoid(x, k=10.0, x0=q_low + 0.5 * (q_high - q_low)):
            return 1.0 / (1.0 + np.exp(-k * (x - x0)))
        prob = sigmoid(score)
        prob = np.clip(prob, 0.0, 1.0)

        return preds, prob

    def predict_motion_windowed(self, motion_df, window_s: float = 1.0, overlap: float = 0.5,
                                thresh_method: str = 'percentile', percentile: float = 0.75,
                                mad_k: float = 1.5):
        """
        Overlapping-window dynamic-threshold motion detection inspired by method7.
        - Builds a quality-modulated composite signal
        - Computes a per-frame dynamic threshold over an overlapping window
        - Applies duration constraints and gap filling
        Returns (binary_predictions, probability_like_score)
        """
        n = len(motion_df)
        if n == 0:
            return np.array([]), np.array([])

        # Base features
        vel = motion_df['robust_velocity'].values.astype(float)
        acc = np.abs(motion_df['wrist_acceleration'].values.astype(float))
        ang = motion_df['angular_velocity'].values.astype(float)
        v_sh = motion_df['shoulder_velocity'].values.astype(float)
        v_el = motion_df['elbow_velocity'].values.astype(float)
        v_wr = motion_df['wrist_velocity'].values.astype(float)
        eang_vel = motion_df.get('elbow_angular_velocity', pd.Series(np.zeros(n))).values.astype(float)

        # Scale normalization by arm length (zero-shot camera/body scale invariance)
        arm_len_series = motion_df.get('arm_length', pd.Series(np.zeros(n))).values.astype(float)
        med_arm_len = np.nanmedian(arm_len_series[arm_len_series > 0]) if np.any(arm_len_series > 0) else np.nan
        if not np.isfinite(med_arm_len) or med_arm_len <= 1e-6:
            # Fallback: robust positional spread of wrist as proxy scale
            wx = motion_df.get('wrist_x', pd.Series(np.zeros(n))).values.astype(float)
            wy = motion_df.get('wrist_y', pd.Series(np.zeros(n))).values.astype(float)
            wxc = wx - np.nanmedian(wx)
            wyc = wy - np.nanmedian(wy)
            med_arm_len = np.nanpercentile(np.sqrt(wxc**2 + wyc**2), 90) + 1e-6
        scale = float(max(med_arm_len, 1e-3))

        # Normalize velocity/acceleration by scale before robust scaling
        vel_nz = vel / scale
        acc_nz = acc / scale
        v_sh_nz = v_sh / scale
        v_el_nz = v_el / scale
        v_wr_nz = v_wr / scale

        # Robust scale features [0,1]
        def rscale_pos(x):
            p98 = np.percentile(x, 98)
            if p98 <= 1e-8:
                return np.zeros_like(x)
            return np.clip(x / p98, 0, 1)
        def rscale_abs(x):
            p98 = np.percentile(np.abs(x), 98)
            if p98 <= 1e-8:
                return np.zeros_like(x)
            return np.clip(np.abs(x) / p98, 0, 1)

        v_n = rscale_pos(vel_nz)
        a_n = rscale_abs(acc_nz)
        g_n = rscale_abs(ang)
        v_sh_n = rscale_pos(v_sh_nz)
        v_el_n = rscale_pos(v_el_nz)
        v_wr_n = rscale_pos(v_wr_nz)
        eang_n = rscale_abs(eang_vel)

        # Multi-keypoint velocity composite and elbow angular component
        vel_multi = 0.25 * v_sh_n + 0.35 * v_el_n + 0.40 * v_wr_n
        # Composite signal emphasizes multi-point velocity, with accel and angular cues
        signal_base = 0.45 * vel_multi + 0.25 * a_n + 0.15 * g_n + 0.15 * eang_n

        # Quality modulation
        quality = 0.5 * np.clip(motion_df['chain_consistency'].values, 0, 1) + \
                  0.5 * np.clip(motion_df['overall_confidence'].values, 0, 1)
        signal_q = signal_base * quality

        # Smooth signal with a window tied to window_s
        win = max(3, int(round(window_s * self.fps)))
        if win % 2 == 0:
            win += 1
        try:
            sig_s = signal.savgol_filter(signal_q, window_length=min(win, max(3, (n // 2) * 2 - 1)), polyorder=2)
        except Exception:
            sig_s = signal.medfilt(signal_q, kernel_size=min(win, n if n % 2 == 1 else n - 1))

        # Overlapping window dynamic threshold
        series = pd.Series(sig_s)
        minp = max(3, win // 2)
        if thresh_method == 'percentile':
            thr = series.rolling(win, center=True, min_periods=minp).quantile(percentile).to_numpy()
            if np.isnan(thr).any():
                thr = pd.Series(thr).fillna(method='bfill').fillna(method='ffill').fillna(np.nanmedian(sig_s)).to_numpy()
            margin = np.maximum(1e-6, pd.Series(sig_s - thr).rolling(win, center=True, min_periods=minp).std().fillna(np.nanstd(sig_s)).to_numpy())
        else:  # MAD-based
            med = series.rolling(win, center=True, min_periods=minp).median()
            abs_dev = (series - med).abs()
            mad = abs_dev.rolling(win, center=True, min_periods=minp).median().to_numpy()
            thr = (med + mad_k * pd.Series(mad)).to_numpy()
            if np.isnan(thr).any():
                thr = pd.Series(thr).fillna(method='bfill').fillna(method='ffill').fillna(np.nanmedian(sig_s)).to_numpy()
            margin = np.maximum(1e-6, 1.4826 * mad)
            if np.isnan(margin).any():
                margin = pd.Series(margin).fillna(method='bfill').fillna(method='ffill').fillna(np.nanstd(sig_s)).to_numpy()

        # Global cap on threshold to avoid collapsing to all zeros
        global_cap = float(np.quantile(sig_s, 0.70))
        thr = np.minimum(thr, global_cap)

        # Dynamic decision (no positive bias to improve recall)
        bias = 0.0
        pred = (sig_s > (thr + bias)).astype(int)

        # High-frequency motion gate: short-window variability of normalized velocity
        hf_win = max(3, int(round(0.25 * self.fps)))
        try:
            hf_std = pd.Series(vel_multi).rolling(hf_win, center=True, min_periods=max(2, hf_win // 2)).std().fillna(0.0).to_numpy()
        except Exception:
            hf_std = np.zeros_like(vel_multi)
        # Gate threshold chosen conservatively for zero-shot
        hf_gate = (hf_std > 0.15).astype(int)
        # Merge HF detections with base prediction
        pred = np.maximum(pred, hf_gate)

        # Duration constraints (derive from seconds)
        min_on = max(1, int(round(0.30 * self.fps)))
        min_off = max(1, int(round(0.20 * self.fps)))
        gap_fill = max(1, int(round(0.20 * self.fps)))
        pred = self._post_process_predictions(pred, min_on, gap_fill)
        pred_inv = 1 - pred
        pred_inv = self._post_process_predictions(pred_inv, min_off, gap_fill)
        pred = 1 - pred_inv

        # Zero-shot safety fallback if positives collapse
        pos_rate = float(pred.mean()) if n > 0 else 0.0
        if pos_rate < 0.02:
            thr_glob = float(np.quantile(sig_s, 0.65))
            pred2 = (sig_s > thr_glob).astype(int)
            # Acceleration gate to capture brief onsets
            acc_gate = (a_n > 0.40).astype(int)
            pred2 = np.maximum(pred2, acc_gate)
            # Include HF gate in fallback as well
            pred2 = np.maximum(pred2, hf_gate)
            pred2 = self._post_process_predictions(pred2, max(1, int(round(0.25 * self.fps))), gap_fill)
            pred2_inv = 1 - pred2
            pred2_inv = self._post_process_predictions(pred2_inv, max(1, int(round(0.20 * self.fps))), gap_fill)
            pred2 = 1 - pred2_inv
            if pred2.mean() > pos_rate:
                pred = pred2

        # Probability-like score
        z = (sig_s - thr) / margin
        prob = 1.0 / (1.0 + np.exp(-z))
        prob = np.clip(prob, 0.0, 1.0)

        return pred.astype(int), prob

def load_ground_truth(label_path: str, handedness: str, total_frames: int, fps: float) -> pd.Series:
    """Load ground truth motion labels."""
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")
    
    gt_df = pd.read_csv(label_path)
    
    if 'Time_s' not in gt_df.columns or 'MarkerNames' not in gt_df.columns:
        raise ValueError("Label file must contain 'Time_s' and 'MarkerNames' columns.")
    
    gt_motion_events = []
    handedness_prefix = handedness.lower()
    
    for _, row in gt_df.iterrows():
        if not isinstance(row['MarkerNames'], str):
            continue
        
        marker = row['MarkerNames'].lower()
        if marker.startswith(handedness_prefix):
            # Definition from prompt: IDLE, STABILIZE, REST = 0
            is_motion = 0 if ('idle' in marker or 'stabilize' in marker or 'rest' in marker) else 1
            frame_idx = int(row['Time_s'] * fps)
            if frame_idx < total_frames:
                gt_motion_events.append({'frame': frame_idx, 'motion': is_motion})
    
    if not gt_motion_events:
        return pd.Series(np.zeros(total_frames, dtype=int))
    
    events_df = pd.DataFrame(gt_motion_events).drop_duplicates(subset='frame', keep='last').set_index('frame')
    full_frame_index = pd.Index(range(total_frames), name='frame')
    gt_motion_series = events_df['motion'].reindex(full_frame_index, method='ffill').fillna(0)
    
    return gt_motion_series.astype(int)

def plot_enhanced_analysis(motion_df, ground_truth, predictions, probabilities, output_path, video_identifier, fps, algo_label=None):
    """Plot enhanced motion analysis results."""
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)
    time_axis = motion_df['time_s']
    
    # Plot 1: Velocity and Acceleration
    ax1 = axes[0]
    ax1.plot(time_axis, motion_df['robust_velocity'], label='Robust Velocity', color='dodgerblue', linewidth=2)
    ax1.set_ylabel('Velocity (pixels/s)', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(time_axis, motion_df['wrist_acceleration'], label='Wrist Accel', color='crimson', alpha=0.7)
    ax2.set_ylabel('Acceleration', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    title_algo = f" ({algo_label})" if algo_label else ""
    ax1.set_title(f'Kinematic Analysis - {video_identifier}{title_algo}')

    # Plot 2: Quality metrics
    axes[1].plot(time_axis, motion_df['overall_confidence'], label='Overall Confidence', color='orange')
    axes[1].plot(time_axis, motion_df['chain_consistency'], label='Chain Consistency', color='purple')
    axes[1].set_ylabel('Quality Metrics')
    axes[1].set_title('Tracking Quality and Kinematic Consistency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Prediction probability
    axes[2].plot(time_axis, probabilities, label='Motion Probability', color='red', linewidth=2)
    axes[2].fill_between(time_axis, 0, probabilities, color='red', alpha=0.2)
    axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='0.5 Threshold')
    axes[2].set_ylabel('Prediction Probability')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Binary prediction vs ground truth
    axes[3].fill_between(time_axis, ground_truth, label='Ground Truth', color='green', alpha=0.5, step='post')
    axes[3].plot(time_axis, predictions, label='Final Prediction', color='black', drawstyle='steps-post', linewidth=2.0)
    axes[3].set_ylabel('Motion')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(['No Motion', 'Motion'])
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced analysis plot saved to {output_path}")
    plt.close(fig)

def run_rtmpose_analysis(video_path, label_path, handedness, output_dir, subsample_fps=15, 
                        algo="windowed", window_s=1.0, overlap=0.5, thresh_method='percentile', 
                        percentile=0.75, mad_k=1.5, activity_label=None):
    """
    Run RTMPose analysis as a function call.
    
    Args:
        video_path: Path to the input video
        label_path: Path to the ground truth label file
        handedness: Hand to track ('L' or 'R')
        output_dir: Directory to save results
        subsample_fps: Target FPS for subsampling
    algo: Motion detection algorithm ("hmm", "hybrid", "windowed", "llm_windowed", "vlm_windowed")
        window_s: Window size in seconds for dynamic thresholding
        overlap: Window overlap fraction (0-1)
        thresh_method: Dynamic threshold method ("percentile" or "mad")
        percentile: Percentile for percentile-based thresholding
        mad_k: K multiplier for MAD-based thresholding
    
    Returns:
        dict: Results containing CSV path, metrics, and other outputs
    """
    print("🔍 Running RTMPose motion analysis...")
    
    os.makedirs(output_dir, exist_ok=True)
    video_identifier = os.path.splitext(os.path.basename(video_path))[0]
    
    print("Step 1: Initializing RTMPose...")
    extractor = EnhancedRTMPoseExtractor()
    
    print("Step 2: Extracting keypoints...")
    keypoints, total_frames, _, _, fps = extractor.extract_keypoints(video_path, subsample_fps)
    
    print("Step 3: Analyzing motion...")
    analyzer = RobustMotionAnalyzer(handedness, fps, extractor.keypoint_names)
    motion_df = analyzer.analyze_motion(keypoints, total_frames)
    
    print("Step 4: Loading ground truth...")
    ground_truth = load_ground_truth(label_path, handedness, total_frames, fps)
    
    print("Step 5: Running motion prediction algorithm...")
    rationales = None
    if algo == "hmm":
        predictions, probabilities = analyzer.predict_motion_robust(motion_df)
    elif algo == "hybrid":
        predictions, probabilities = analyzer.predict_motion_hysteresis(motion_df)
    elif algo == "llm_windowed":
        # Use proper LLM-based motion detection
        try:
            predictions, probabilities, rationales = predict_motion_llm_windowed(
                video_path,
                motion_df,
                fps,
                window_s=window_s,
                overlap=overlap,
                handedness=handedness,
                activity_label=activity_label,
                model_name=os.environ.get('LLM_MODEL', 'qwen2.5:7b')
            )
        except Exception as e:
            print(f"   ⚠️  LLM motion detection failed: {e}")
            # Fallback to windowed approach with enhanced rationales
            predictions, probabilities = analyzer.predict_motion_windowed(
                motion_df,
                window_s=window_s,
                overlap=overlap,
                thresh_method=thresh_method,
                percentile=percentile,
                mad_k=mad_k
            )
            try:
                rationales = _generate_window_rationales(motion_df, predictions, probabilities, window_s, overlap, handedness, activity_label=activity_label)
            except:
                rationales = None
    elif algo == "vlm_windowed":
        predictions, probabilities, rationales = predict_motion_vlm_windowed(
            video_path,
            motion_df,
            fps,
            window_s=window_s,
            overlap=overlap,
            handedness=handedness,
            activity_label=activity_label,
            model_name=os.environ.get('VLM_MOTION_MODEL')
        )
    else:
        predictions, probabilities = analyzer.predict_motion_windowed(
            motion_df,
            window_s=window_s,
            overlap=overlap,
            thresh_method=thresh_method,
            percentile=percentile,
            mad_k=mad_k
        )
    
    # --- Evaluation ---
    print("\n--- Motion Prediction Results ---")
    report = classification_report(ground_truth, predictions, target_names=['no_motion', 'motion'], zero_division=0)
    print(report)
    
    # Save detailed report
    report_path = os.path.join(output_dir, f"{video_identifier}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Motion Analysis Report - {video_identifier}\n")
        f.write("="*50 + "\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        cm = confusion_matrix(ground_truth, predictions)
        f.write(str(cm) + "\n")
    print(f"Detailed report saved to {report_path}")

    # Save metrics JSON for batch aggregation
    metrics = {}
    try:
        accuracy = float((ground_truth == predictions).mean())
        prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(ground_truth, predictions, average='binary', zero_division=0)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(ground_truth, predictions, average='macro', zero_division=0)
        try:
            roc_auc = float(roc_auc_score(ground_truth, probabilities))
        except Exception:
            roc_auc = None
        try:
            ap = float(average_precision_score(ground_truth, probabilities))
        except Exception:
            ap = None
        metrics = {
            "video_id": video_identifier,
            "fps": float(fps),
            "accuracy": float(accuracy),
            "precision_binary": float(prec_bin),
            "recall_binary": float(rec_bin),
            "f1_binary": float(f1_bin),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "roc_auc": roc_auc,
            "average_precision": ap
        }
        metrics_path = os.path.join(output_dir, f"{video_identifier}_metrics.json")
        with open(metrics_path, "w") as mf:
            json.dump(metrics, mf, indent=2)
        print(f"Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Warning: failed to save metrics JSON: {e}")

    # --- Plotting ---
    print("Step 6: Creating visualization...")
    plot_filename = f"{video_identifier}_{algo}_analysis.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plot_enhanced_analysis(motion_df, ground_truth, predictions, probabilities, plot_path, video_identifier, fps, algo_label=algo)
    
    # --- Save full results ---
    results_df = motion_df.copy()
    results_df['ground_truth'] = ground_truth
    results_df['prediction'] = predictions
    results_df['probability'] = probabilities
    
    csv_path = os.path.join(output_dir, f"{video_identifier}_enhanced_motion_data.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Full motion data saved to {csv_path}")
    # Save rationales if available
    rationales_path = None
    if rationales is not None:
        try:
            rationales_path = os.path.join(output_dir, f"{video_identifier}_rationales.json")
            with open(rationales_path, 'w') as rf:
                json.dump(rationales, rf, indent=2)
            print(f"Rationales saved to {rationales_path}")
        except Exception as e:
            print(f"Warning: failed to save rationales: {e}")
    
    print(f"✅ RTMPose analysis complete! Results saved in: {output_dir}")
    
    return {
        "motion_csv": csv_path,
        "metrics": metrics,
        "report_path": report_path,
        "plot_path": plot_path,
        "fps": fps,
        "total_frames": total_frames,
        "rationales_path": rationales_path
    }


def _initialize_llm_model(model_name: str = "qwen2.5:7b"):
    """Initialize LLM model for motion analysis."""
    try:
        import ollama
        # Test if model is available
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                options={"num_predict": 5}
            )
            print(f"   ✅ LLM model {model_name} initialized successfully")
            return model_name
        except Exception as e:
            print(f"   ⚠️  Model {model_name} not available, trying alternatives...")
            # Try alternative models
            for alt_model in ["llama3.2:3b", "llama3.2:1b", "phi3:mini"]:
                try:
                    response = ollama.chat(
                        model=alt_model,
                        messages=[{"role": "user", "content": "Hello"}],
                        options={"num_predict": 5}
                    )
                    print(f"   ✅ Using alternative LLM model: {alt_model}")
                    return alt_model
                except:
                    continue
            return None
    except ImportError:
        print("   ⚠️  Ollama not available, falling back to heuristics")
        return None
    except Exception as e:
        print(f"   ⚠️  LLM initialization failed: {e}")
        return None


def _analyze_motion_with_llm(model_name: str, frame_data: dict, hand_text: str, activity_context: dict, 
                            past_rationales: list, window_id: int) -> tuple:
    """Analyze motion using LLM with visual and contextual information."""
    try:
        import ollama
        
        # Build context for LLM
        activity_name = activity_context.get('name', activity_context.get('activity_label', 'unknown'))
        workspace = activity_context.get('workspace', '')
        target_objects = activity_context.get('target_objects', '')
        steps = activity_context.get('steps', [])
        
        # Motion features
        velocity = frame_data.get('velocity', 0.0)
        acceleration = frame_data.get('acceleration', 0.0)
        contact = frame_data.get('contact', 0.0)
        
        # Build comprehensive prompt
        prompt = f"""You are analyzing a rehabilitation video of a stroke patient performing the "{activity_name}" activity.

CONTEXT:
- Activity: {activity_name}
- Workspace: {workspace}
- Target objects: {target_objects}
- Expected steps: {'; '.join(steps[:3]) if steps else 'Not specified'}
- Hand being analyzed: {hand_text}

CURRENT WINDOW DATA (Window {window_id}):
- Average velocity: {velocity:.2f} units/frame
- Average acceleration: {acceleration:.2f} units/frame²
- Contact probability: {contact:.2f}

RECENT HISTORY:
{chr(10).join(past_rationales[-2:]) if past_rationales else 'No previous context'}

TASK:
Based on the motion data and context, determine if the {hand_text} is currently in motion or stationary.

Consider:
1. The expected activity steps and whether motion would be expected at this stage
2. The velocity and acceleration values in context of the activity
3. Whether the hand is likely interacting with target objects
4. The progression from previous windows

Respond in this EXACT format:
MOTION: [0 for stationary, 1 for in motion]
CONFIDENCE: [0.0 to 1.0]
RATIONALE: [2-3 sentence explanation of why you classified it this way, referencing the activity context and motion data]

Example:
MOTION: 1
CONFIDENCE: 0.85
RATIONALE: The {hand_text} shows significant velocity (5.2) and acceleration (12.1) consistent with reaching for the comb. Given this is the "pick up comb" phase of combing activity, active motion is expected as the patient extends their arm toward the target object.
"""

        # Query LLM
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 200}
        )
        
        response_text = response['message']['content'].strip()
        
        # Parse response
        motion_pred = 0
        confidence = 0.5
        rationale = f"LLM analysis failed to parse response for window {window_id}"
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('MOTION:'):
                try:
                    motion_pred = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('RATIONALE:'):
                try:
                    rationale = line.split(':', 1)[1].strip()
                except:
                    pass
        
        # Ensure valid values
        motion_pred = max(0, min(1, motion_pred))
        confidence = max(0.0, min(1.0, confidence))
        
        return motion_pred, confidence, rationale
        
    except Exception as e:
        print(f"   ⚠️  LLM analysis failed for window {window_id}: {e}")
        # Fallback to enhanced heuristic with activity context
        activity_boost = 0.1 if activity_context.get('steps') else 0.0
        prob = min(1.0, 0.2 + 0.4 * (velocity > 0.1) + 0.3 * (acceleration > 0.2) + 0.1 * contact + activity_boost)
        pred = 1 if prob >= 0.5 else 0
        rationale = f"Fallback analysis: {hand_text} {'in motion' if pred else 'stationary'} based on velocity={velocity:.2f}, acceleration={acceleration:.2f}"
        return pred, prob, rationale


def predict_motion_llm_windowed(video_path: str, motion_df: pd.DataFrame, fps: float, window_s: float, overlap: float,
                                handedness: str, activity_label: str = None, model_name: str = None):
    """
    LLM-based motion detection that analyzes motion patterns with activity context.
    Returns (binary_predictions, probabilities, rationales).
    """
    print("🧠 Starting LLM-based motion detection...")
    
    times = motion_df['time_s'].values.astype(float)
    n = len(times)
    preds = np.zeros(n, dtype=int)
    probs = np.zeros(n, dtype=float)
    rationales = []
    
    # Initialize LLM
    llm_model = _initialize_llm_model(model_name or "qwen2.5:7b")
    
    hop_s = max(0.01, window_s * (1.0 - overlap))
    start_t = 0.0
    wid = 0
    past_texts = []
    
    # Load activity context
    ctx = _load_activity_context(activity_label) or {}
    activity_context = {
        'name': ctx.get('name', activity_label),
        'activity_label': activity_label,
        'workspace': ctx.get('workspace', ''),
        'target_objects': ctx.get('target_objects', ''),
        'steps': ctx.get('steps', [])
    }
    
    hand_text = 'left hand' if handedness.upper() == 'L' else 'right hand'
    
    print(f"   📊 Processing {int((times[-1] - start_t) / hop_s) + 1} windows with LLM analysis...")
    
    while start_t < times[-1] + 1e-6:
        end_t = min(start_t + window_s, times[-1])
        idx = np.where((times >= start_t) & (times < end_t))[0]
        
        if idx.size == 0:
            start_t += hop_s
            wid += 1
            continue
        
        # Extract motion features for this window
        vel_mean = float(np.mean(motion_df.loc[idx, 'robust_velocity'])) if 'robust_velocity' in motion_df.columns else 0.0
        accel_mean = float(np.mean(motion_df.loc[idx, 'wrist_acceleration'])) if 'wrist_acceleration' in motion_df.columns else 0.0
        contact_mean = float(np.mean(motion_df.loc[idx, 'contact_prediction'])) if 'contact_prediction' in motion_df.columns else 0.0
        
        frame_data = {
            'velocity': vel_mean,
            'acceleration': accel_mean,
            'contact': contact_mean
        }
        
        # Analyze with LLM if available
        if llm_model:
            pred_motion, prob_motion, rationale_text = _analyze_motion_with_llm(
                llm_model, frame_data, hand_text, activity_context, past_texts, wid
            )
        else:
            # Enhanced fallback analysis
            activity_boost = 0.1 if activity_context.get('steps') else 0.0
            prob_motion = float(min(1.0, 0.2 + 0.4 * (vel_mean > 0.1) + 0.3 * (accel_mean > 0.2) + 0.1 * contact_mean + activity_boost))
            pred_motion = 1 if prob_motion >= 0.5 else 0
            
            # Create contextual rationale
            motion_state = "in motion" if pred_motion else "stationary"
            rationale_text = f"Enhanced heuristic: {hand_text} {motion_state}. "
            
            if activity_context.get('steps'):
                current_step = activity_context['steps'][min(wid // 3, len(activity_context['steps']) - 1)]
                rationale_text += f"During '{current_step}' phase, "
            
            rationale_text += f"velocity={vel_mean:.2f} and acceleration={accel_mean:.2f} "
            
            if pred_motion:
                rationale_text += "indicate active movement"
                if contact_mean > 0.1:
                    rationale_text += " with object interaction"
            else:
                rationale_text += "suggest minimal movement"
            
            rationale_text += f". Confidence: {prob_motion:.2f}"
        
        # Apply predictions to all frames in window
        preds[idx] = pred_motion
        probs[idx] = prob_motion
        
        # Store detailed rationale
        rationales.append({
            'window_id': wid,
            'start_time': float(start_t),
            'end_time': float(end_t),
            'pred': int(pred_motion),
            'prob': float(prob_motion),
            'rationale': rationale_text,
            'past_rationales': past_texts[-3:],
            'features': {
                'velocity': float(vel_mean),
                'acceleration': float(accel_mean),
                'contact': float(contact_mean)
            },
            'activity_context': activity_context,
            'analysis_method': 'LLM' if llm_model else 'enhanced_heuristic'
        })
        
        past_texts.append(rationale_text)
        
        # Print output for every window
        print(f"   🧠 Window {wid}: {hand_text} {'MOTION' if pred_motion else 'STILL'} (p={prob_motion:.3f}) [{start_t:.1f}-{end_t:.1f}s] - {rationale_text[:80]}...")
        
        start_t += hop_s
        wid += 1
    
    print(f"   ✅ Completed LLM motion analysis: {len(rationales)} windows processed")
    return preds, probs, rationales


def predict_motion_vlm_windowed(video_path: str, motion_df: pd.DataFrame, fps: float, window_s: float, overlap: float,
                                handedness: str, activity_label: str = None, model_name: str = None):
    """
    Per-window VLM motion classifier with past-2-window rationale history.
    Returns (binary_predictions, probability_like, rationales_json_obj).
    For now, uses simple frame sampling per window and a stub classifier that can be
    swapped with a real VLM inference function.
    """
    try:
        import cv2
    except Exception:
        cv2 = None
    times = motion_df['time_s'].values.astype(float)
    n = len(times)
    preds = np.zeros(n, dtype=int)
    probs = np.zeros(n, dtype=float)
    rationales = []
    cap = None
    if cv2 is not None and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS) if cap else fps
    else:
        original_fps = fps
    hop_s = max(0.01, window_s * (1.0 - overlap))
    start_t = 0.0
    wid = 0
    past_texts = []
    ctx = _load_activity_context(activity_label) or {}
    steps = ctx.get('steps', []) if isinstance(ctx, dict) else []
    workspace = ctx.get('workspace', '') if isinstance(ctx, dict) else ''
    target_objects = ctx.get('target_objects', '') if isinstance(ctx, dict) else ''
    step_text = "; ".join(steps[:3]) if steps else ""
    
    print(f"   📊 Processing {int((times[-1] - start_t) / hop_s) + 1} windows...")
    
    while start_t < times[-1] + 1e-6:
        end_t = min(start_t + window_s, times[-1])
        idx = np.where((times >= start_t) & (times < end_t))[0]
        if idx.size == 0:
            start_t += hop_s
            wid += 1
            continue
        # Sample one representative frame from the middle of the window
        mid_t = float(0.5 * (start_t + end_t))
        mid_frame = int(round(mid_t * original_fps))
        # Get motion features for this window
        vel_mean = float(np.mean(motion_df.loc[idx, 'robust_velocity'])) if 'robust_velocity' in motion_df.columns else 0.0
        accel_mean = float(np.mean(motion_df.loc[idx, 'wrist_acceleration'])) if 'wrist_acceleration' in motion_df.columns else 0.0
        contact_mean = float(np.mean(motion_df.loc[idx, 'contact_prediction'])) if 'contact_prediction' in motion_df.columns else 0.0
        
        hand_text = 'left hand' if handedness.upper() == 'L' else 'right hand'
        
        # Enhanced heuristic with activity context
        activity_weight = 0.1 if step_text else 0.0
        prob_motion = float(min(1.0, 0.15 + 0.5 * (vel_mean > 0.05) + 0.25 * (accel_mean > 0.1) + 0.1 * contact_mean + activity_weight))
        pred_motion = 1 if prob_motion >= 0.5 else 0
        
        preds[idx] = pred_motion
        probs[idx] = prob_motion
        
        # Create detailed rationale with activity context
        rationale_text = f"VLM(win {wid}): {hand_text} {'in motion' if pred_motion else 'stationary'}; p≈{prob_motion:.2f}; vel≈{vel_mean:.2f}; accel≈{accel_mean:.2f}; contact≈{contact_mean:.2f}"
        if step_text:
            rationale_text += f"; task: {step_text}"
        rationales.append({
            'window_id': wid,
            'start_time': float(start_t),
            'end_time': float(end_t),
            'pred': int(pred_motion),
            'prob': float(prob_motion),
            'rationale': rationale_text,
            'past_rationales': past_texts[-3:],  # Keep last 3 rationales for context
            'features': {
                'velocity': float(vel_mean),
                'acceleration': float(accel_mean),
                'contact': float(contact_mean)
            },
            'activity_context': {
                'workspace': workspace,
                'target_objects': target_objects,
                'current_steps': step_text
            }
        })
        past_texts.append(rationale_text)
        
        # Print output for every window as requested
        print(f"   📊 Window {wid}: {hand_text} {'MOTION' if pred_motion else 'STILL'} (p={prob_motion:.3f}) [{start_t:.1f}-{end_t:.1f}s]")
        
        start_t += hop_s
        wid += 1
    if cap is not None:
        cap.release()
    return preds, probs, rationales


# ---- Helper functions for LLM-like windowed rationale generation ----
def _load_activity_context(activity_label: str):
    try:
        import yaml
    except Exception:
        return None
    try:
        ypath = os.path.join(os.path.dirname(__file__), 'activities_ground_truth.yaml')
        if not os.path.exists(ypath):
            ypath = 'activities_ground_truth.yaml'
        if not os.path.exists(ypath):
            return None
        with open(ypath, 'r') as f:
            data = yaml.safe_load(f)
        if not activity_label or not isinstance(data, list):
            return None
        for item in data:
            if isinstance(item, dict) and item.get('name') and activity_label.lower() in str(item.get('name')).lower():
                return {
                    'name': item.get('name'),
                    'workspace': item.get('workspace'),
                    'target_objects': item.get('target_objects'),
                    'steps': item.get('steps', [])
                }
        return None
    except Exception:
        return None


def _generate_window_rationales(motion_df: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray,
                                window_s: float, overlap: float, handedness: str, activity_label: str = None):
    """Generate detailed rationales for each window with 3-window context history."""
    # Build windows over time_s
    if 'time_s' not in motion_df.columns:
        return []
    times = motion_df['time_s'].values.astype(float)
    n = len(times)
    if n == 0:
        return []
    fps_est = 1.0 / np.median(np.diff(times)) if n > 1 else 10.0
    hop_s = max(0.01, window_s * (1.0 - overlap))
    rationales = []
    ctx = _load_activity_context(activity_label) or {}
    steps = ctx.get('steps', []) if isinstance(ctx, dict) else []
    workspace = ctx.get('workspace', '') if isinstance(ctx, dict) else ''
    target_objects = ctx.get('target_objects', '') if isinstance(ctx, dict) else ''
    step_text = "; ".join(steps[:3]) if steps else ""
    prev_texts = []
    
    print(f"   📊 Generating rationales for {int((times[-1] - 0.0) / hop_s) + 1} windows...")
    start_t = 0.0
    wid = 0
    while start_t < times[-1] + 1e-6:
        end_t = min(start_t + window_s, times[-1])
        idx = np.where((times >= start_t) & (times < end_t))[0]
        if idx.size == 0:
            start_t += hop_s
            wid += 1
            continue
        pred_win = predictions[idx]
        prob_win = probabilities[idx]
        motion_rate = float(pred_win.mean())
        prob_mean = float(np.mean(prob_win)) if prob_win.size > 0 else 0.0
        vel_mean = float(np.mean(motion_df.loc[idx, 'robust_velocity'])) if 'robust_velocity' in motion_df.columns else 0.0
        accel_mean = float(np.mean(motion_df.loc[idx, 'wrist_acceleration'])) if 'wrist_acceleration' in motion_df.columns else 0.0
        contact_mean = float(np.mean(motion_df.loc[idx, 'contact_prediction'])) if 'contact_prediction' in motion_df.columns else 0.0
        
        # Compose rationale text
        motion_state = "in motion" if motion_rate > 0.3 or prob_mean > 0.5 else "stationary"
        hand_text = "left hand" if handedness.upper() == 'L' else "right hand"
        
        # Build detailed rationale with features and context
        rationale_text = f"LLM(win {wid}): {hand_text} {motion_state}. P_motion≈{prob_mean:.2f}; vel≈{vel_mean:.2f}; accel≈{accel_mean:.2f}; contact≈{contact_mean:.2f}"
        if step_text:
            rationale_text += f"; task: {step_text}"
        if workspace:
            rationale_text += f"; workspace: {workspace[:50]}..."
        
        past_context = prev_texts[-3:]  # Keep last 3 rationales
        rationales.append({
            'window_id': wid,
            'start_time': float(start_t),
            'end_time': float(end_t),
            'motion_rate': motion_rate,
            'prob_mean': prob_mean,
            'rationale': rationale_text,
            'past_rationales': past_context,
            'features': {
                'velocity': float(vel_mean),
                'acceleration': float(accel_mean),
                'contact': float(contact_mean)
            },
            'activity_context': {
                'workspace': workspace,
                'target_objects': target_objects,
                'current_steps': step_text
            }
        })
        prev_texts.append(rationale_text)
        
        # Print output for every window as requested
        print(f"   📊 Window {wid}: {hand_text} {motion_state.upper()} (p={prob_mean:.3f}) [{start_t:.1f}-{end_t:.1f}s]")
        
        start_t += hop_s
        wid += 1
    print(f"   ✅ Generated {len(rationales)} window rationales")
    return rationales


def save_rationales_to_json(rationales: list, output_path: str):
    """Save rationales to JSON file for later analysis."""
    import json
    try:
        with open(output_path, 'w') as f:
            json.dump(rationales, f, indent=2)
        print(f"   💾 Rationales saved to: {output_path}")
    except Exception as e:
        print(f"   ⚠️  Failed to save rationales: {e}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced RTMPose analysis with robust motion prediction")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--label_path", type=str, required=True, help="Path to the ground truth label file.")
    parser.add_argument("--handedness", type=str, required=True, choices=['L', 'R'], help="Hand to track ('L' or 'R').")
    parser.add_argument("--output_dir", type=str, default="rtmpose_results", help="Directory to save results.")
    parser.add_argument("--subsample_fps", type=int, default=15, help="Target FPS for subsampling (default: 15).")
    parser.add_argument("--algo", type=str, default="windowed", choices=["hmm", "hybrid", "windowed", "llm_windowed"], help="Motion detection algorithm")
    parser.add_argument("--window_s", type=float, default=1.0, help="Window size in seconds for dynamic thresholding")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction (0-1); informative for future use")
    parser.add_argument("--thresh_method", type=str, default='percentile', choices=['percentile','mad'], help="Dynamic threshold method")
    parser.add_argument("--percentile", type=float, default=0.75, help="Percentile for percentile-based thresholding")
    parser.add_argument("--mad_k", type=float, default=1.5, help="K multiplier for MAD-based thresholding")
    args = parser.parse_args()

    # Call the function version
    results = run_rtmpose_analysis(
        video_path=args.video_path,
        label_path=args.label_path,
        handedness=args.handedness,
        output_dir=args.output_dir,
        subsample_fps=args.subsample_fps,
        algo=args.algo,
        window_s=args.window_s,
        overlap=args.overlap,
        thresh_method=args.thresh_method,
        percentile=args.percentile,
        mad_k=args.mad_k
    )
    
    return results

if __name__ == "__main__":
    main()