import os, glob, copy, cv2, numpy as np, torch, torch.nn as nn
from tqdm import tqdm

# your repo imports
from tools.motionagformer.lib.utils import normalize_screen_coordinates, camera_to_world
from tools.motionagformer.model.MotionAGFormer import MotionAGFormer

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --------------------------
# Fake 2D keypoints generator
# --------------------------
def fake_2d_keypoints(video_path: str, use_conf: bool = True) -> tuple[np.ndarray, float, int, int]:
    """
    Pretend 2D keypoints for a single person in H36M order.
    Returns:
      kpts2d: (T, 17, 3) if usef_conf else (T, 17, 2), pixel coordinates
      fps, width, height (from the video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Center and simple walking-ish sway
    cx, cy = width * 0.5, height * 0.55
    t = np.arange(T, dtype=np.float32)
    sway = 0.05 * width * np.sin(2 * np.pi * t / max(1, (fps * 1.2)))

    # A very simple stick figure in H36M order (rough proportions, pixels)
    # 0: pelvis/root, 1-3: L hip/knee/ankle, 4-6: R hip/knee/ankle,
    # 7: spine, 8: thorax, 9: nose/head base, 10: head top-ish,
    # 11-13: R shoulder/elbow/wrist, 14-16: L shoulder/elbow/wrist
    base = np.array([
        [  0,   0],    # 0  pelvis
        [  0,  40],    # 1  L hip
        [  0,  90],    # 2  L knee
        [  0, 140],    # 3  L ankle
        [  0,  40],    # 4  R hip
        [  0,  90],    # 5  R knee
        [  0, 140],    # 6  R ankle
        [  0, -35],    # 7  spine mid
        [  0, -75],    # 8  thorax
        [  0, -105],   # 9  head base
        [  0, -135],   # 10 head top
        [ 32, -70],    # 11 R shoulder
        [ 55, -40],    # 12 R elbow
        [ 70, -15],    # 13 R wrist
        [-32, -70],    # 14 L shoulder
        [-55, -40],    # 15 L elbow
        [-70, -15],    # 16 L wrist
    ], dtype=np.float32)

    kpts2d = np.zeros((T, 17, 3 if use_conf else 2), dtype=np.float32)
    rng = np.random.default_rng(0)

    for i in range(T):
        # gentle sway + tiny jitter
        off = np.array([cx + sway[i], cy], dtype=np.float32)
        jitter = rng.normal(0, 1.0, size=base.shape).astype(np.float32)
        xy = base + jitter + off  # (17, 2)

        # clamp to frame
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)

        if use_conf:
            conf = np.full((17, 1), 0.9, dtype=np.float32)
            kpts2d[i] = np.concatenate([xy, conf], axis=-1)
        else:
            kpts2d[i] = xy

    cap.release()
    return kpts2d, float(fps), int(width), int(height)

# --------------------------
# 3D prediction using fake 2D
# --------------------------
@torch.no_grad()
def run_3d_from_fake_2d(video_path: str, out_npz_path: str, gpu: str = "0", use_conf: bool = True):
    """
    Generates fake 2D pose, runs MotionAGFormer to get 3D, saves NPZ:
      times: (T,), kpts2d: (T,17,3/2), kpts3d: (T,17,3)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    kpts2d, fps, width, height = fake_2d_keypoints(video_path, use_conf=use_conf)
    T = kpts2d.shape[0]
    times = np.arange(T, dtype=np.float32) / fps

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
        if args["dim_in"] == 3:
            if use_conf and clip.shape[-1] == 3:
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

        out_nonflip = model(t_in)               # (1,243,17,3)
        out_flip    = flip_poses(model(t_aug))  # (1,243,17,3)
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
        meta=dict(fps=float(fps), width=int(width), height=int(height), use_conf=bool(use_conf)),
    )
    print(f"Saved: {out_npz_path}")


# --- skeleton wiring shared with your runner ---
H36M_CONNECTIONS = np.array([
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
    [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]
], dtype=int)
LR_2D = np.array([0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0], dtype=bool)
LR_3D = np.array([0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,0], dtype=bool)

def _draw_2d_pose_on_image(kps_2d, img):
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

def _render_3d_pose_image(kps_3d, radius_xy=0.72, radius_z=0.70, fig_size=(4.8, 4.8)):
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 1); gs.update(wspace=0.0, hspace=0.0)
    ax = plt.subplot(gs[0], projection='3d')
    ax.view_init(elev=15., azim=70)

    for i, (a, b) in enumerate(H36M_CONNECTIONS):
        x, y, z = kps_3d[[a, b], 0], kps_3d[[a, b], 1], kps_3d[[a, b], 2]
        ax.plot(x, y, z, lw=2, color=(0,0,1) if LR_3D[i] else (1,0,0))

    x0, y0, z0 = kps_3d[0]
    ax.set_xlim3d([x0 - radius_xy, x0 + radius_xy])
    ax.set_ylim3d([y0 - radius_xy, y0 + radius_xy])
    ax.set_zlim3d([z0 - radius_z,  z0 + radius_z])
    ax.set_aspect('auto')
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1.0,1.0,1.0,0.0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Matplotlib >= 3.6+ friendly path
    try:
        # Returns an RGBA memoryview; zero-copy into (H,W,4)
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        img = rgba[..., :3].copy()  # drop alpha
    except AttributeError:
        # Fallback for older MPL that only has tostring_argb()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = argb[..., 1:4].copy()  # ARGB -> RGB

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
    # FYI see https://github.com/TaatiTeam/MotionAGFormer/blob/master/requirements.txt
    # pip install timm==0.6.11

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)  # /gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_combing1_1.mkv
    ap.add_argument("--npz_out", required=True)  # to_delete.npz
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--no_conf", action="store_true")
    args = ap.parse_args()

    run_3d_from_fake_2d(
        video_path=args.video,
        out_npz_path=args.npz_out,
        gpu=args.gpu,
        use_conf=not args.no_conf,
    )

    pose2video(
        npz_path=args.npz_out,
        original_video_path=args.video,
        out_video_path="to_delete.mp4"
    )
