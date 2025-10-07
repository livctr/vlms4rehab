# StrokeRehab Pipeline

This pipeline operationalizes the full stack from 2D pose lifting through motion segmentation, kinematic reasoning, multimodal contact detection, and primitive synthesis. The codebase targets repeatable, zero-shot deployment on in-the-wild clinical video with frame-level explainability.

## TL;DR
- **Pose**: RTMPose extracts shoulder-elbow-wrist tracks, denoised with a constant-velocity Kalman filter that adapts to detector confidence.
- **Motion**: Composite dynamics fuse multi-joint velocity, acceleration, and elbow angular velocity into an HMM, hysteresis, or overlapping-window decoder tuned for sub-second movements.
- **Kinematics**: We compute robust velocity, jerk, chain consistency, arm length, and elbow angles to ground the motion signal in interpretable biomechanical cues.
- **Contact**: A vision-language model (Qwen2.5-VL or InternVL3) consumes activity context and representative frames to classify hand-object contact over sliding temporal windows.
- **Primitives**: Motion and contact streams are fused into reach/reposition/transport/stabilize/idle primitives with future-look heuristics to respect occupational therapy semantics.

## Repository Layout
Note: Heavy outputs and example results live under `.misc/` and are gitignored for portability.
```
organized_pipeline/
|- main.py                  # End-to-end orchestration across modalities
|- enhanced_rtmpose_analysis.py   # Pose extraction, Kalman smoothing, motion decoding
|- contact_detection_vlm.py       # VLM-based contact estimation with sliding windows
|- primitives_utils.py            # Motion/contact -> primitive logic and metrics
|- enhanced_video_generator.py    # Diagnostic overlay video synthesis
|- batch_process.py               # Metadata-driven batch execution
|- utils.py                       # Label ingestion and handedness helpers
|- vic/                           # Sequence-metric evaluation scripts (AER, Edit)
|- cleaned_metadata.csv / activities_ground_truth.yaml
```

## Data Requirements
Base data path defaults to `/gpfs/data/schambralab/quantitativeRehabilitation/__data/`. You can override via `--base_data_path`.
- **Video**: RGB files at arbitrary frame rates. Subsampling to ~15 FPS is handled internally.
- **Labels (optional)**: CSVs with `Time_s` + `MarkerNames` for evaluation; handedness is inferred if unspecified.
- **Activity context**: `activities_ground_truth.yaml` captures task-specific objects and procedural steps for prompting the VLM.
- **Metadata**: `cleaned_metadata.csv` records per-session paths, handedness overrides, and activity names for batch execution.

## Pipeline Overview
1. `main.py` provisions per-session directories, resolves handedness, and dispatches modality-specific modules.
2. `enhanced_rtmpose_analysis.run_rtmpose_analysis` extracts pose keypoints, smooths trajectories, derives motion cues, and outputs frame-wise predictions plus diagnostics.
3. `contact_detection_vlm.run_contact_detection` aligns motion CSVs with subsampled frames, queries the VLM, and saves window-level contact probabilities.
4. Outputs are synchronized, deduplicated, and converted into primitive sequences; evaluation metrics are computed when reference labels exist.
5. `enhanced_video_generator.EnhancedMotionVideoGenerator` optionally renders interpretability overlays for qualitative review.

## Pose & Kinematic Stack
- **Detector**: RTMPose-X (Halpe26) via `mmpose`. Config and checkpoint paths fall back to bundled artifacts if local files are missing.
- **Temporal Smoothing**: `EnhancedKalmanFilter` models shoulder, elbow, and wrist with a shared constant-velocity state; measurement noise is adapted using detector confidences to down-weight occlusions.
- **Feature Bank**:
  - Shoulder/elbow/wrist velocities and accelerations (Kalman-derived).
  - Robust wrist velocity = velocity * chain consistency * confidence, used as the core motion energy.
  - Chain consistency gauges kinematic plausibility via relative limb segment lengths.
  - Elbow angle, elbow angular velocity, arm length, and composite orientation for biomechanical context.
  - jerk and high-frequency variability features for motion onset sensitivity.
- **Plotting & Reports**: Each run yields motion plots (`*_enhanced_analysis.png`), classification reports, and JSON summaries for aggregation.

## Motion Segmentation Algorithms
We expose three complementary decoders (select via `--algo`):
- `windowed` (default): Overlapping-window dynamic thresholding with quality modulation, high-frequency gating, and duration priors for clinical micro-movements.
- `hmm`: Probabilistic decoding relying on a sigmoid-mapped composite score, Viterbi smoothing, and post-processing to enforce minimum bout durations.
- `hybrid`: Hysteresis thresholds with HSMM-like duration enforcement to accommodate noisy cues while preserving quick onset detection.
All modes share the same composite signal built from multi-joint velocity, acceleration, angular velocity, and elbow dynamics, scaled through robust percentiles.

## Contact Detection Module
- **Windowing**: Motion predictions define sliding windows (`--window_s`, `--overlap`) whose majority state drives motion labels per window.
- **Prompt Assembly**: For each window we construct a JSON prompt embedding activity-specific objects, handedness, motion state, and the timestamp.
- **Model Backends**: Default is `Qwen/Qwen2.5-VL-7B-Instruct`; `OpenGVLab/InternVL3-38B` is supported for higher-capacity runs when multi-GPU memory is available.
- **Vision Pipeline**: Images are dynamically tiled to meet aspect-ratio constraints, normalized with ImageNet statistics, and streamed through transformers with optional multi-GPU sharding.
- **Outputs**: `*_window_contact.csv` stores per-window contact predictions, confidences, and contextual metadata. Representative window clips can be exported for auditing.

## Primitive Synthesis
`primitives_utils.convert_motions_and_contacts_to_prims` aligns the motion/contact sequences over window boundaries and emits action primitives:
- `reach`: Motion before impending contact (look-ahead window default 2 s).
- `reposition`: Motion without forthcoming contact.
- `transport`: Motion while in contact.
- `stabilize`: Static contact segments.
- `idle`: Static, no-contact segments.
Deduplication removes micro-oscillations and ensures contiguous runs match clinical annotations. Metrics (Edit Score, AER) are computed via `primitives_utils.get_primitives_score` against ground truth primitives loaded through `utils.LabelUtils`.

## Outputs
Per session we materialize:
- `motion_analysis/{id}/{id}_enhanced_motion_data.csv` - frame-wise kinematics, motion probabilities, and ground truth if provided.
- `contact_detection/{id}/{id}_window_contact.csv` - window-level contact calls and confidences.
- `final_results/{id}_final_predictions.csv` - synchronized frame-level motion, contact, kinematic features, and primitives.
- Diagnostics: plots, classification reports, optional overlay videos.


## Inference (example)
```bash
python main.py --skip_motion --motion_csv "/gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/naveen/final_pipeline/the_pipeline/strokerehab/strokerehab/organized_pipeline/C00020_glasses1_1_final/motion_analysis/C00020_glasses1_1/C00020_glasses1_1_vlm_motion.csv" --video_id "C00020_glasses1_1" --algo vlm_motion --vlm_motion_model OpenGVLab/InternVL3-38B --model OpenGVLab/InternVL3-38B --motion_window_s 0.25 --motion_overlap 0.1 --contact_mode framewise --contact_frame_fps 15 --contact_batch_size 8 --contact_median_kernel 3 --contact_gaussian_sigma 1.0 --contact_high_threshold 0.7 --contact_low_threshold 0.3 --contact_min_run_frames 3 --contact_gap_fill_frames 2 --clear_cache --low_memory --multi_gpu --output_dir "C00020_glasses1_1_final"
```

Notes:
- --skip_motion uses an existing motion CSV to resume.
- --algo vlm_motion uses VLM for motion; contact is framewise here.
- --multi_gpu with InternVL3 can reduce OOM risk; adjust batch sizes.

## Reproducing Experiments
### Single Session
```bash
python main.py \
  --video_path path/to/video.mp4 \
  --label_path path/to/labels.csv \
  --activity "face wash" \
  --handedness L \
  --output_dir results/face_wash \
  --subsample_fps 15 \
  --algo windowed \
  --window_s 1.0 \
  --overlap 0.5
```

### Batch Sweep
```bash
python batch_process.py \
  --metadata_csv cleaned_metadata.csv \
  --video_base_path /abs/path/to/videos \
  --label_base_path /abs/path/to/labels \
  --output_dir batch_results \
  --n_videos all
```

### Evaluation Only
Reuse an existing motion/contact pair to recompute primitives or metrics:
```bash
python pipeline_summary.py \
  --results_root pipeline_results \
  --video_id S00027_feeding1_1
```

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment & Dependencies
- Python 3.8+
- Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `opencv-python`
- Deep Learning: `torch`, `mmpose`, `transformers`, `qwen-vl-utils`, optional `bitsandbytes`
- Evaluation: `python-Levenshtein`
Install via `pip install -r requirements.txt`. GPU execution is strongly recommended for pose estimation and VLM inference; CPU fallbacks are supported but slow.


