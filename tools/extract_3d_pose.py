# Core and numerical
import os, glob, copy, json
import numpy as np

# Deep learning
import torch
import torch.nn as nn

# Imaging
import cv2

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Progress bar
from tqdm import tqdm

# Type hints
from typing import Any, Dict, List, Optional, Tuple, Union
Number = Union[int, float]

# Local imports
from data.utils_strokerehab import DataPaths
from lmms_eval.tasks.strokerehab.utils_primitives import load_strokerehab_primitives_dataset
from lmms_eval.tasks.strokerehab.utils_impairment_analysis import load_strokerehab_ia_dataset
from tools.motionagformer.lib.utils import normalize_screen_coordinates, camera_to_world
from tools.motionagformer.model.MotionAGFormer import MotionAGFormer
from tools.ultralytics_pose import Pose2DStream
# from tools.hrnet.gen_kpts import Pose2DStream
from tools.vqa.qwen2_5_vl import Qwen2_5_VL_VQA


def _to_float4(b: Union[List[Number], Tuple[Number, Number, Number, Number]]) -> Tuple[float, float, float, float]:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError(f"Expected bbox of length 4, got {type(b).__name__} with len={len(b) if hasattr(b, '__len__') else 'N/A'}")
    x1, y1, x2, y2 = b
    for v in (x1, y1, x2, y2):
        if not isinstance(v, (int, float)):
            raise TypeError(f"BBox elements must be numbers, got {type(v).__name__}")
    return float(x1), float(y1), float(x2), float(y2)

def _strip_to_json_array(s: str) -> str:
    # Robustly extract the JSON array substring, even if surrounded by code fences
    s = s.strip()
    start = s.find('[')
    end = s.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a JSON array in the input string.")
    return s[start:end+1]

def extract_first_bbox_and_label(
    detections: Union[str, List[Dict[str, Any]]]
) -> Tuple[Tuple[float, float, float, float], Optional[str]]:
    """
    Returns (bbox, label) ready to pass into:
        add_new_person_to_track(bbox=bbox, label=label)

    - Accepts a JSON string (optionally within code fences) OR a parsed Python list.
    - Ensures bbox is (x1, y1, x2, y2) as floats.
    """
    if isinstance(detections, str):
        detections_json = _strip_to_json_array(detections)
        detections_list: List[Dict[str, Any]] = json.loads(detections_json)
    elif isinstance(detections, list):
        detections_list = detections
    else:
        raise TypeError(f"Unsupported type for detections: {type(detections).__name__}")

    if not detections_list:
        raise ValueError("No detections found.")

    first = detections_list[0]
    if not isinstance(first, dict):
        raise TypeError("Each detection must be a dict.")

    # Support 'bbox_2d' (your example) and fallback to 'bbox'
    if "bbox_2d" in first:
        raw_bbox = first["bbox_2d"]
    elif "bbox" in first:
        raw_bbox = first["bbox"]
    else:
        raise KeyError("Detection missing 'bbox_2d' (or 'bbox') key.")

    bbox = _to_float4(raw_bbox)
    label = first.get("label")
    return bbox, label


LOCATE_PATIENT_PROMPT = (
    "Locate the person performing the activity as a bounding box in JSON."
)

def extract_2d_pose(video_path: str, vqa_model: Qwen2_5_VL_VQA, stream_2d: Pose2DStream):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
    
        if frame_idx == 0:
            patient_loc_text = vqa_model.process_frames(frame, context=LOCATE_PATIENT_PROMPT)
            bbox, label = extract_first_bbox_and_label(patient_loc_text)
            stream_2d.add_new_person_to_track(bbox=bbox, label=label)

        stream_2d.process_frame(frame)
        frame_idx += 1
    cap.release()

    all_kpts = stream_2d.get()  # (T, 1, 17, 3)
    all_kpts = all_kpts[:, 0, :, :]  # (T, 17, 3)
    return all_kpts, fps, width, height


def smooth_pose2d_kalman(
    pose_xyz: np.ndarray,
    *,
    fps: float = 30.0,
    process_accel_var: float = 1e4,
    meas_var_base: float = 1.0,
    conf_floor: float = 5-2,
    do_rts_smoothing: bool = False,
    smooth_conf_alpha: float = 0.25,
) -> np.ndarray:
    """
    Kalman smoother for (T, 17, 3) arrays of (x, y, conf) from 2D pose estimation.

    Model: Constant velocity per joint with white-acceleration process noise.
      state s = [x, y, vx, vy]^T
      s_{t+1} = A s_t + w_t,    w_t ~ N(0, Q)
      z_t     = H s_t + v_t,    v_t ~ N(0, R_t), with R_t scaled by 1/conf_t

    Args:
        pose_xyz: np.ndarray of shape (T, 17, 3) with (x, y, conf).
                  Missing measurements can be given as NaNs or conf=0.
        fps: Frames per second (sets Δt).
        process_accel_var: Process noise spectral density (px^2 / s^4). Larger = smoother, slower to react.
        meas_var_base: Baseline per-axis measurement variance (px^2) when conf=1.0.
        conf_floor: Minimum confidence used to avoid division by zero.
        do_rts_smoothing: If True, apply an RTS backward pass for smoother trajectories.
        smooth_conf_alpha: EMA factor to gently smooth the confidence channel (0=no smoothing).

    Returns:
        smoothed: np.ndarray of shape (T, 17, 3) with smoothed (x, y, conf).
    """
    T, J, C = pose_xyz.shape
    assert C == 3, "Expected last dimension to be (x, y, conf)."
    dt = 1.0 / float(fps)

    # State-space matrices (shared across joints)
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=float)

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    # Process noise Q from continuous white-acceleration model
    dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
    q = process_accel_var
    Q = q * np.array([
        [dt4/4,    0.0,   dt3/2,  0.0],
        [0.0,    dt4/4,   0.0,   dt3/2],
        [dt3/2,    0.0,   dt2,    0.0],
        [0.0,    dt3/2,   0.0,    dt2]
    ], dtype=float)

    # Storage
    x_filt = np.zeros((T, J, 4), dtype=float)  # filtered (or smoothed later) state means
    P_filt = np.zeros((T, J, 4, 4), dtype=float)  # covariances

    # Initialize per-joint state/cov
    for j in range(J):
        x0, y0, c0 = pose_xyz[0, j]
        # If the first frame is missing, try to find the first valid frame
        if (not np.isfinite(x0)) or (not np.isfinite(y0)) or (c0 <= 0):
            valid = np.where(np.isfinite(pose_xyz[:, j, 0]) &
                             np.isfinite(pose_xyz[:, j, 1]) &
                             (pose_xyz[:, j, 2] > 0))[0]
            if len(valid) == 0:
                # No measurements at all: keep zeros with large uncertainty
                x0 = y0 = 0.0
            else:
                x0, y0 = pose_xyz[valid[0], j, :2]

        # Start velocity at zero, large uncertainty in both position & velocity
        x_filt[0, j] = np.array([x0, y0, 0.0, 0.0], dtype=float)
        P_filt[0, j] = np.diag([1e4, 1e4, 1e4, 1e4])

    # Forward pass (Kalman filter)
    for t in range(1, T):
        for j in range(J):
            # Predict
            x_pred = A @ x_filt[t-1, j]
            P_pred = A @ P_filt[t-1, j] @ A.T + Q

            zx, zy, zc = pose_xyz[t, j]
            have_meas = np.isfinite(zx) and np.isfinite(zy) and (zc > 0.0)

            if have_meas:
                conf = float(np.clip(zc, conf_floor, 1.0))
                # Measurement noise grows when confidence is low
                R = (meas_var_base / conf) * np.eye(2)
                z = np.array([zx, zy], dtype=float)

                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                y = z - H @ x_pred
                x_upd = x_pred + K @ y
                P_upd = (np.eye(4) - K @ H) @ P_pred
            else:
                # No update if measurement is missing
                x_upd, P_upd = x_pred, P_pred

            x_filt[t, j] = x_upd
            P_filt[t, j] = P_upd

    # Optional RTS smoother (backward pass)
    if do_rts_smoothing:
        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)

        for t in range(T - 2, -1, -1):
            for j in range(J):
                P_t = P_filt[t, j]
                P_tp1 = P_filt[t+1, j]
                # Predict step from t to t+1 (recompute, cheap)
                x_pred = A @ x_filt[t, j]
                P_pred = A @ P_t @ A.T + Q

                # Smoother gain
                C = P_t @ A.T @ np.linalg.inv(P_pred)
                # Update
                x_smooth[t, j] = x_filt[t, j] + C @ (x_smooth[t+1, j] - x_pred)
                P_smooth[t, j] = P_t + C @ (P_smooth[t+1, j] - P_pred) @ C.T

        states = x_smooth
    else:
        states = x_filt

    # Extract (x, y)
    xy_smooth = states[..., :2]  # (T, 17, 2)

    # Gently smooth confidence with a causal EMA (independent of KF math)
    conf = pose_xyz[..., 2]
    conf_s = np.copy(conf)
    if smooth_conf_alpha > 0.0:
        for j in range(J):
            c_prev = conf_s[0, j]
            for t in range(1, T):
                # If missing conf, treat as zero to reflect uncertainty
                c_t = conf[t, j] if np.isfinite(conf[t, j]) else 0.0
                c_prev = smooth_conf_alpha * c_t + (1 - smooth_conf_alpha) * c_prev
                conf_s[t, j] = c_prev
        # Clip to [0,1]
        conf_s = np.clip(conf_s, 0.0, 1.0)

    # Pack output
    out = np.zeros_like(pose_xyz, dtype=float)
    out[..., 0:2] = xy_smooth
    out[..., 2] = conf_s
    return out


def get_motionagformer_model_and_args():
    args = dict(
        n_layers=26, dim_in=3, dim_feat=128, dim_rep=512, dim_out=3,
        mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.0, drop=0.0, drop_path=0.0,
        use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
        num_heads=8, qkv_bias=False, qkv_scale=None, hierarchical=False,
        use_temporal_similarity=True, neighbour_num=2, temporal_connection_len=1,
        use_tcn=False, graph_only=False, n_frames=243
    )
    model = nn.DataParallel(MotionAGFormer(**args)).cuda()
    ckpt = sorted(glob.glob(os.path.join("tools", "motionagformer", "checkpoint", "motionagformer-l-h36m.pth.tr")))
    if not ckpt:
        raise FileNotFoundError("Checkpoint not found at checkpoint/motionagformer-l-h36m.pth.tr*")
    state = torch.load(ckpt[0], map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, args

@torch.no_grad()
def run_3d_pose_extraction(
    video_path: str,
    out_npz_path: str,
    vqa_model: Qwen2_5_VL_VQA,
    stream_2d: Pose2DStream,
    motionagformer_model: nn.Module,
    motionagformer_args: Dict[str, Any]
):
    """
    Generates 2D pose, runs MotionAGFormer to get 3D, saves NPZ:
      times: (T,), kpts2d: (T,17,3/2), kpts3d: (T,17,3)
    """

    kpts2d, fps, width, height = extract_2d_pose(video_path, vqa_model, stream_2d)
    kpts2d = turn_into_h36m(kpts2d)
    # kpts2d = smooth_pose2d_kalman(kpts2d, fps=fps)

    T = kpts2d.shape[0]
    # times = np.arange(T, dtype=np.float32) / fps

    # Build clip plan of length 243 windows
    def _resample_indices(n_frames, target=243):
        even = np.linspace(0, n_frames, num=target, endpoint=False)
        idx = np.floor(even)
        return np.clip(idx, 0, n_frames - 1).astype(np.uint32)

    def make_clip_plan(n_frames, window=243):
        plan = []
        if n_frames <= window:
            new_idx = _resample_indices(n_frames, window)
            uniq, _ = np.unique(new_idx, return_index=True)
            plan.append((new_idx, uniq))
            return plan
        for start in range(0, n_frames, window):
            L = min(window, n_frames - start)
            if L == window:
                feed = np.arange(start, start + window, dtype=np.uint32)
                place = feed
            else:
                new_idx = _resample_indices(L, window)
                feed = start + new_idx
                place = start + np.unique(new_idx)
            plan.append((feed, place))
        return plan

    plan = make_clip_plan(T, window=243)
    kpts3d = np.zeros((T, 17, 3), dtype=np.float32)

    # Predict 3D
    for feed_idx, place_idx in tqdm(plan, desc="3D"):
        clip = kpts2d[feed_idx]          # (243,17,2/3)
        clip_xy = clip[..., :2]          # (243,17,2)

        # normalize to [-1,1]
        clip_xy_norm = normalize_screen_coordinates(clip_xy[None, ...], w=width, h=height)  # (1,243,17,2)

        # ensure model gets 3 channels
        if motionagformer_args["dim_in"] == 3:
            if clip.shape[-1] == 3:
                conf = clip[..., 2:3]  # (243,17,1)
            else:
                conf = np.ones_like(clip_xy[..., :1], dtype=np.float32)
            inp = np.concatenate([clip_xy_norm, conf[None, ...]], axis=-1)  # (1,243,17,3)
        else:
            inp = clip_xy_norm  # (1,243,17,2)

        # simple flip TTA (flip x and swap left/right limbs)
        def flip_poses(data, left=[1,2,3,14,15,16], right=[4,5,6,11,12,13]):
            out = copy.deepcopy(data)
            out[..., 0] *= -1
            out[..., left + right, :] = out[..., right + left, :]
            return out

        t_in  = torch.from_numpy(inp.astype(np.float32)).cuda()
        t_aug = torch.from_numpy(flip_poses(inp).astype(np.float32)).cuda()

        out_nonflip = motionagformer_model(t_in)               # (1,243,17,3)
        out_flip    = flip_poses(motionagformer_model(t_aug))  # (1,243,17,3)
        out_3d = (out_nonflip + out_flip) / 2   # (1,243,17,3)

        # tail handling: select only unique placements when resampled
        if len(place_idx) < 243:
            start = feed_idx.min()
            within = feed_idx - start
            uniq, first = np.unique(within, return_index=True)
            placed = out_3d[0, first].detach().cpu().numpy()
        else:
            placed = out_3d[0].detach().cpu().numpy()

        # zero pelvis x (match your original)
        placed[:, 0, 0] = 0.0

        # camera_to_world + depth shift + scale
        rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
                       dtype=np.float32)
        for i_f, g in enumerate(place_idx):
            p = camera_to_world(placed[i_f], R=rot, t=0)
            p[:, 2] -= np.min(p[:, 2])
            m = np.max(p)
            if m > 0:
                p /= m
            kpts3d[g] = p.astype(np.float32)

    # Save NPZ
    np.savez_compressed(
        out_npz_path,
        times=(np.arange(T, dtype=np.float32) / fps),
        kpts2d=kpts2d.astype(np.float32),
        kpts3d=kpts3d.astype(np.float32),
        meta=dict(fps=float(fps), width=int(width), height=int(height)),
    )
    print(f"Saved: {out_npz_path}")


# --- skeleton wiring shared with your runner ---
H36M_CONNECTIONS = np.array([
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
    [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]
], dtype=int)
LR_2D = np.array([0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0], dtype=bool)
LR_3D = np.array([0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0], dtype=bool)


def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]
    return new_keypoints

def _draw_2d_pose_on_image(kps_2d, img):
    # kps_2d = turn_into_h36m(kps_2d)
    lcolor = (255, 0, 0)  # BGR
    rcolor = (0, 0, 255)
    thickness = 3
    for j, (a, b) in enumerate(H36M_CONNECTIONS):
        p1 = tuple(map(int, kps_2d[a, :2]))
        p2 = tuple(map(int, kps_2d[b, :2]))
        cv2.line(img, p1, p2, lcolor if LR_2D[j] else rcolor, thickness)
        cv2.circle(img, p1, 3, (0,255,0), -1)
        cv2.circle(img, p2, 3, (0,255,0), -1)
    return img

def _render_3d_pose_image(
    kps_3d,
    radius_xy=0.72,
    radius_z=0.70,
    fig_size=(8, 8)
):
    """
    Render a 2x2 panel of the same 3D skeleton from 4 different viewpoints.
    Returns a NumPy RGB image.
    """
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(2, 2); gs.update(wspace=0.0, hspace=0.0)

    # Pick four viewpoints (elev, azim)
    views = [
        (15,  0),   # default-ish
        (15,  90),   # mirrored azimuth
        (15,  180),  # from the back
        (15,  270),   # top-down
    ]

    for i, (elev, azim) in enumerate(views):
        ax = plt.subplot(gs[i], projection="3d")
        ax.view_init(elev=elev, azim=azim)

        for j, (a, b) in enumerate(H36M_CONNECTIONS):
            x, y, z = kps_3d[[a, b], 0], kps_3d[[a, b], 1], kps_3d[[a, b], 2]
            ax.plot(x, y, z, lw=2, color=(0, 0, 1) if LR_3D[j] else (1, 0, 0))

        x0, y0, z0 = kps_3d[0]
        ax.set_xlim3d([x0 - radius_xy, x0 + radius_xy])
        ax.set_ylim3d([y0 - radius_xy, y0 + radius_xy])
        ax.set_zlim3d([z0 - radius_z,  z0 + radius_z])
        ax.set_aspect("auto")
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    try:
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        img = rgba[..., :3].copy()
    except AttributeError:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = argb[..., 1:4].copy()

    plt.close(fig)
    return img


def pose2video(npz_path, original_video_path, out_video_path, three_d_figsize=(4.8, 4.8)):
    """
    Left: original frame with 2D keypoints overlay
    Right: Matplotlib-rendered 3D skeleton for the same frame
    """
    data = np.load(npz_path, allow_pickle=True)
    k2d = data["kpts2d"]   # (T, 17, 2 or 3)
    k3d = data["kpts3d"]   # (T, 17, 3)

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {original_video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T_vid  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = min(T_vid, k2d.shape[0], k3d.shape[0])

    # Prepare right panel size based on first render (square-ish)
    sample_right = _render_3d_pose_image(k3d[0], fig_size=three_d_figsize)
    r_h, r_w = sample_right.shape[:2]
    scale = height / r_h
    out_right_w = max(1, int(r_w * scale))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width + out_right_w, height))

    print("Rendering side-by-side video…")
    for t in tqdm(range(T)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Overlay 2D on the original frame
        frame_drawn = _draw_2d_pose_on_image(k2d[t], frame.copy())

        # Render 3D to RGB, convert to BGR, resize to match height
        right = _render_3d_pose_image(k3d[t], fig_size=three_d_figsize)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        right = cv2.resize(right, (out_right_w, height), interpolation=cv2.INTER_AREA)

        writer.write(np.concatenate([frame_drawn, right], axis=1))

    writer.release()
    cap.release()
    print(f"Wrote: {out_video_path}")





# --------------------------
# EXAMPLE USAGE
# --------------------------
if __name__ == "__main__":

    HRNET_CFG = "tools/hrnet/experiments/w48_384x288_adam_lr1e-3.yaml"
    HRNET_WEIGHTS = "tools/hrnet/checkpoint/pose_hrnet_w48_384x288.pth"

    video_regex = r'^(C00020/C00020_glasses1_1.mkv|C00020/C00020_drinking1_1.mkv|C00020/C00020_combing1_1.mkv|C00020/C00020_face wash1_1.mkv|C00020/C00020_shelf right side1_1.mkv|C00020/C00020_deodrant1_1.mkv)$'
    ex_paths = load_strokerehab_primitives_dataset(video_regex=video_regex)['test']['path_v']

    prims_input_dir = DataPaths.RAW_VIDEO_DIR
    prims_paths = load_strokerehab_primitives_dataset(filter_for_subsampled_testset=True)['test']['path_v']
    prims_output_dir = DataPaths.ADL_POSE_DIR

    ia_input_dir = DataPaths.IA_CLIPPED_VIDEO_DIR
    ia_paths = load_strokerehab_ia_dataset(metadata_path=DataPaths.IA_VIDEO_METADATA_PATH3)['test']['path_v']
    ia_output_dir = DataPaths.FM_POSE_DIR

    vqa_model = Qwen2_5_VL_VQA(pretrained="Qwen/Qwen2.5-VL-32B-Instruct", device="cuda:0", device_map="auto")
    stream_2d = Pose2DStream(num_person=1)

    motionagformer_model, motionagformer_args = get_motionagformer_model_and_args()

    for input_dir, paths, output_dir in [
        (prims_input_dir, ex_paths, prims_output_dir),
        # (prims_input_dir, prims_paths, prims_output_dir),
        # (ia_input_dir, ia_paths, ia_output_dir),
    ]:
        os.makedirs(output_dir, exist_ok=True)
        for p in paths:
            video_path = os.path.join(input_dir, p)
            basename = os.path.basename(p).split(".")[0]
            out_path = os.path.join(output_dir, basename + ".npz")
            if os.path.exists(out_path):
                print(f"Skipping existing: {out_path}")
                continue
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print(f"Processing: {video_path} -> {out_path}")

            vqa_model.clear()
            stream_2d.clear()
            try:
                run_3d_pose_extraction(
                    video_path=video_path,
                    out_npz_path=out_path,
                    vqa_model=vqa_model,
                    stream_2d=stream_2d,
                    motionagformer_model=motionagformer_model,
                    motionagformer_args=motionagformer_args
                )
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue

    pose2video(
        npz_path=out_path,
        original_video_path=video_path,
        out_video_path="to_delete.mp4"
    )
