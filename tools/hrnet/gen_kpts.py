from __future__ import annotations

import copy
import os
import types
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# HRNet utilities
from tools.hrnet.lib.config import cfg, update_config
from tools.hrnet.lib.models import pose_hrnet
from tools.hrnet.lib.utils.inference import get_final_preds
from tools.hrnet.lib.utils.utilitys import PreProcess

# YOLO human detector + SORT tracker
from tools.sort.sort import Sort
from tools.yolov3.human_detector import load_model as yolo_model
from tools.yolov3.human_detector import yolo_human_det as yolo_det

class Pose2DStream:
    def __init__(
        self,
        # --- mirror original parse_args defaults ---
        cfg_path: str = "tools/hrnet/experiments/w48_384x288_adam_lr1e-3.yaml",  # --cfg
        modelDir: str = "tools/hrnet/checkpoint/pose_hrnet_w48_384x288.pth",     # --modelDir
        det_dim: int = 416,                       # --det-dim
        thred_score: float = 0.30,                # --thred-score
        animation: bool = False,                  # -a / --animation
        num_person: int = 1,                      # -np / --num-person
        video: str = "camera",                    # -v / --video
        gpu: str = "0",                           # --gpu
        # extra: opts lets you override YAML keys like the CLI `opts`
        opts: list[str] | None = None,
        # extra: explicit device override
        device: str | None = None,
        # extra: whether SORT should include detection scores in its output
        sort_include_score: bool = True,
    ):
        """
        Initializes models and tracker. All relevant arguments are passed into update_config(cfg, args_like)
        to mirror the original CLI-driven setup, without argparse.
        """
        # Make sure CUDA_VISIBLE_DEVICES follows the user's gpu selection (if any)
        if gpu is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))

        # --- Build an args-like object that matches HRNet's expectations ---
        # Include every relevant attribute from the original script.
        args_like = types.SimpleNamespace(
            cfg=cfg_path,
            opts=list(opts or []),
            modelDir=modelDir,
            det_dim=det_dim,
            thred_score=thred_score,
            animation=bool(animation),
            num_person=int(num_person),
            video=video,
            gpu=str(gpu),
            # Common extras update_config accesses in HRNet repos:
            modelDir1="", logDir="", dataDir="", outputDir="", prevModelDir="",
        )

        # Let HRNet merge YAML + opts + any side effects it performs
        update_config(cfg, args_like)

        # cuDNN settings follow cfg (like reset_config in the script)
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        # Device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Persist key runtime params (used elsewhere in the class)
        self.num_person = int(num_person)
        self.yolo_inp_dim = int(det_dim)
        self.det_conf_threshold = float(thred_score)

        # Load detector / pose / tracker
        self.detector = yolo_model(inp_dim=self.yolo_inp_dim)
        self.pose_model = self._load_hrnet(weights_path=modelDir)
        self.tracker = Sort(min_hits=0)

        # Stream state (initialize as your class uses)
        self._prev_bboxs = None
        self._prev_scores = None
        self._frames = []
        self._last_tracks = None

        self.sort_include_score = sort_include_score

        # Slots: list of dicts per person slot
        self._slots: List[Dict] = [
            {"track_id": None, "label": None, "anchor": None, "last_bbox": None}
            for _ in range(self.num_person)
        ]
        # Pending prompts (center,label) to bind when tracks are next available
        self._pending_prompts: List[Tuple[Tuple[float, float], Optional[str]]] = []

    def _load_hrnet(self, weights_path: str | None = None):
        """
        Build HRNet from cfg and load weights.

        Resolution order for weights:
          1) Explicit `weights_path` (modelDir from init).
          2) cfg.MODEL.PRETRAINED (common in HRNet repos).
          3) cfg.OUTPUT_DIR if it points to a file.
        """
        model = pose_hrnet.get_pose_net(cfg, is_train=False).to(self.device)

        # Resolve a usable checkpoint path
        candidate_paths = []
        if weights_path:
            candidate_paths.append(weights_path)
        # Some HRNet configs carry a pretrained path here
        try:
            pretrained = getattr(cfg.MODEL, "PRETRAINED", "")
            if pretrained:
                candidate_paths.append(pretrained)
        except Exception:
            pass
        # Some forks stick a direct file path into OUTPUT_DIR
        try:
            outdir = getattr(cfg, "OUTPUT_DIR", "")
            if outdir and os.path.isfile(outdir):
                candidate_paths.append(outdir)
        except Exception:
            pass

        ckpt_path = next((p for p in candidate_paths if isinstance(p, str) and os.path.isfile(p)), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Could not find HRNet weights. Tried: {candidate_paths or ['<none provided>']}"
            )

        # Load state dict (support both plain and {'state_dict': ...})
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)

        # Strip a possible 'module.' prefix (DataParallel)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                          for k, v in state_dict.items()}

        # Load (be strict, but fall back to non-strict if needed)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model

    def _prepare_dets_for_sort(self, bboxs: np.ndarray, scores: np.ndarray | None) -> np.ndarray:
        """
        Ensure SORT gets detections in the expected shape.
        Returns an array shaped (N, 5): [x1,y1,x2,y2,score]
        If scores is None, we fabricate score=1.0 for each box.
        """
        if bboxs is None or len(bboxs) == 0:
            return np.empty((0, 5), dtype=np.float32)

        bboxs = np.asarray(bboxs, dtype=np.float32)
        if bboxs.shape[1] >= 5:
            # already has scores (YOLO sometimes returns Nx6 with class, etc. — keep only score as 5th col)
            return np.column_stack([bboxs[:, 0:4], bboxs[:, 4].astype(np.float32)])

        if scores is None:
            scores_col = np.ones((bboxs.shape[0], 1), dtype=np.float32)
        else:
            s = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
            scores_col = s if s.shape[0] == bboxs.shape[0] else np.ones((bboxs.shape[0], 1), dtype=np.float32)

        return np.hstack([bboxs[:, 0:4], scores_col])
    
    def _nearest_track(self, center_xy: Tuple[float, float], tracks: np.ndarray) -> Tuple[int, Tuple[float, float, float, float]]:
        if tracks is None or tracks.size == 0:
            raise ValueError("No tracks to select from.")
        x, y = center_xy
        boxes = tracks[:, :4]
        ids = tracks[:, -1].astype(int)  # id is always last col (Nx5 or Nx6)
        ctrs = np.column_stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5))
        idx = int(np.argmin((ctrs[:, 0] - x) ** 2 + (ctrs[:, 1] - y) ** 2))
        return ids[idx], tuple(map(float, boxes[idx]))

    # --------- Prompting API (SAM-style) ---------
    def add_new_person_to_track(
        self,
        point: Optional[Tuple[float, float]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        label: Optional[str] = None,
    ) -> int:
        """
        Bind the closest current SORT track to a free slot based on a prompt.

        Args:
            point: (x, y) in pixel coords.
            bbox: (x1, y1, x2, y2) in pixel coords.
            label: optional user label for this tracked person.

        Returns:
            slot_idx (0..M-1) that was bound.

        Behavior:
            - If tracks exist now, bind immediately to nearest track by center distance.
            - If no tracks yet, store as pending; it will auto-bind on the next frame with tracks.
            - Raises ValueError if all slots are already used.
        """
        if point is None and bbox is None:
            raise ValueError("Provide either point=(x,y) or bbox=(x1,y1,x2,y2).")
        if point is not None and bbox is not None:
            raise ValueError("Provide only one of point or bbox, not both.")

        center = point if point is not None else ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

        # Find a free slot
        free_slots = [i for i, s in enumerate(self._slots) if s["track_id"] is None and s["anchor"] is None]
        if not free_slots:
            raise ValueError("All slots are already bound. Increase num_person or clear a slot.")

        slot_idx = free_slots[0]

        # If we already have tracks, bind now; otherwise, defer
        if self._last_tracks is not None and self._last_tracks.shape[0] > 0:
            track_id, track_bbox = self._nearest_track(center, self._last_tracks)
            self._slots[slot_idx].update(
                {"track_id": int(track_id), "label": label, "anchor": center, "last_bbox": track_bbox}
            )
        else:
            self._slots[slot_idx].update({"track_id": None, "label": label, "anchor": center, "last_bbox": None})
            self._pending_prompts.append((center, label))

        return slot_idx

    def _bind_pending_prompts(self, tracks: np.ndarray):
        """
        Bind any pending prompts to the nearest available tracks (does not dedupe already-bound ids).
        """
        if not self._pending_prompts or tracks is None or tracks.shape[0] == 0:
            return
        for center, label in list(self._pending_prompts):
            # find a free slot
            free_slots = [i for i, s in enumerate(self._slots) if s["track_id"] is None]
            if not free_slots:
                break
            slot_idx = free_slots[0]
            track_id, track_bbox = self._nearest_track(center, tracks)
            self._slots[slot_idx].update(
                {"track_id": int(track_id), "label": label, "anchor": center, "last_bbox": track_bbox}
            )
            self._pending_prompts.pop(0)

    # -------------- Core streaming --------------
    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a fixed-shape array (1, self.num_person, 17, 3), using np.nan where data is unavailable.
        """
        K = 17
        # Default output: all NaNs so downstream can mask with np.isnan
        out = np.full((1, self.num_person, K, 3), np.nan, dtype=np.float32)

        # 1) Detect
        bboxs, scores = yolo_det(
            frame_bgr, self.detector, reso=self.yolo_inp_dim, confidence=self.det_conf_threshold
        )

        if bboxs is None or (hasattr(bboxs, "any") and not bboxs.any()):
            # fall back to previous detector outputs if available
            if self._prev_bboxs is not None and len(self._prev_bboxs) > 0:
                bboxs, scores = self._prev_bboxs, self._prev_scores
            else:
                # no detections and no fallback → record NaN frame and return
                self._frames.append(out)
                return out
        else:
            self._prev_bboxs = copy.deepcopy(bboxs)
            self._prev_scores = copy.deepcopy(scores)

        # Prepare dets for SORT (always Nx5 = [x1,y1,x2,y2,score])
        dets_for_sort = self._prepare_dets_for_sort(bboxs, scores)

        # 2) Track — request 5- or 6-wide output depending on flag
        tracks = self.tracker.update(dets_for_sort, include_score=self.sort_include_score)  # Nx(5 or 6)
        self._last_tracks = tracks.copy() if tracks is not None and tracks.size else None

        if tracks is None or tracks.shape[0] == 0:
            # no tracks → record NaN frame
            self._frames.append(out)
            return out

        # Bind pending prompts if any
        self._bind_pending_prompts(tracks)

        # Build selection of up to self.num_person boxes, prioritizing bound slots
        last_col = tracks.shape[1] - 1
        id_to_box = {int(t[last_col]): tuple(map(float, t[:4])) for t in tracks}
        used_ids = set()
        selected_boxes: List[Tuple[float, float, float, float]] = []

        # 2a) Fill with bound slots first
        for s in self._slots:
            box = None
            if s["track_id"] is not None and s["track_id"] in id_to_box:
                box = id_to_box[s["track_id"]]
                s["last_bbox"] = box
                used_ids.add(s["track_id"])
            elif s["last_bbox"] is not None:
                box = s["last_bbox"]
            if box is not None:
                selected_boxes.append(tuple(round(float(v), 2) for v in box))
            if len(selected_boxes) >= self.num_person:
                break

        # 2b) Fill remaining with any other current tracks
        if len(selected_boxes) < self.num_person:
            for tid, box in id_to_box.items():
                if tid in used_ids:
                    continue
                selected_boxes.append(tuple(round(float(v), 2) for v in box))
                used_ids.add(tid)
                if len(selected_boxes) >= self.num_person:
                    break

        # 2c) If still short, fall back to previous detector boxes
        if len(selected_boxes) < self.num_person and self._prev_bboxs is not None and len(self._prev_bboxs) > 0:
            for i in range(self._prev_bboxs.shape[0]):
                b = tuple(map(float, self._prev_bboxs[i, :4]))
                selected_boxes.append(tuple(round(v, 2) for v in b))
                if len(selected_boxes) >= self.num_person:
                    break

        # If nothing usable this frame, keep NaNs and return
        if len(selected_boxes) == 0:
            self._frames.append(out)
            return out

        # Effective count for this frame
        m_eff = min(len(selected_boxes), self.num_person)
        selected_boxes = selected_boxes[:m_eff]

        # 3) HRNet on the boxes we actually have
        inputs, origin_img, center, scale = PreProcess(frame_bgr, selected_boxes, cfg, m_eff)
        inputs = inputs[:, [2, 1, 0]].to(self.device)
        output = self.pose_model(inputs)
        preds, maxvals = get_final_preds(cfg, output.detach().cpu().numpy(), np.asarray(center), np.asarray(scale))

        # 4) Fill the fixed output (NaNs already placed elsewhere)
        out[0, :m_eff, :, :2] = preds[:m_eff].astype(np.float32)
        conf = maxvals.squeeze().astype(np.float32)
        conf = conf if conf.ndim == 2 else conf.reshape(m_eff, K)
        out[0, :m_eff, :, 2] = conf

        # Update slot last bboxes only for available slots
        for i, s in enumerate(self._slots[:m_eff]):
            s["last_bbox"] = selected_boxes[i]

        self._frames.append(out)
        return out


    def process_video(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self.reset_results()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.process_frame(frame)
        cap.release()
        return self.get()

    def reset_results(self):
        self._frames.clear()

    # Retrieval
    def get(self, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        if not self._frames:
            return np.zeros((0, self.num_person, 17, 3), dtype=np.float32)
        full = np.concatenate(self._frames, axis=0)
        return full if (start is None and end is None) else full[slice(start, end)]

    def __getitem__(self, sl: Union[int, slice, Sequence[int]]) -> np.ndarray:
        return self.get()[sl]

    def __len__(self) -> int:
        if not self._frames:
            return 0
        return np.concatenate(self._frames, axis=0).shape[0]

    # Optional helpers
    def get_slot_labels(self) -> List[Optional[str]]:
        return [s["label"] for s in self._slots]

    def clear_slot(self, slot_idx: int):
        self._slots[slot_idx] = {"track_id": None, "label": None, "anchor": None, "last_bbox": None}

    def slots_info(self) -> List[Dict]:
        """Debug/inspection: [{'track_id':..., 'label':..., 'anchor':..., 'last_bbox':...}, ...]"""
        return copy.deepcopy(self._slots)

    def clear(
        self,
        *,
        keep_slot_labels: bool = False,
        keep_pending_prompts: bool = False
    ) -> None:
        """
        Reset streamer state so you can run on a new video.

        Args:
            keep_slot_labels: if True, preserve user-provided labels in the slots;
                            only track_id/anchor/last_bbox are cleared.
            keep_pending_prompts: if True, keep any prompts that haven't been bound yet.
        """
        # 1) Per-video caches/buffers
        self.reset_results()               # clears self._frames
        self._prev_bboxs = None
        self._prev_scores = None
        self._last_tracks = None

        # 2) Tracker state: fresh SORT instance
        # (recreate with the same settings you used in __init__)
        self.tracker = Sort(min_hits=0)

        # 3) Slots
        if keep_slot_labels:
            for s in self._slots:
                s["track_id"] = None
                s["anchor"] = None
                s["last_bbox"] = None
            # labels preserved
        else:
            for i in range(len(self._slots)):
                self._slots[i] = {"track_id": None, "label": None, "anchor": None, "last_bbox": None}

        # 4) Pending prompts
        if not keep_pending_prompts:
            self._pending_prompts.clear()
