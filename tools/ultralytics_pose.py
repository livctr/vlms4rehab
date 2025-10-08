from __future__ import annotations

# --- NEW imports (top of file) ---
import hashlib
import pickle
from pathlib import Path

import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tools.sort.sort import Sort


class Pose2DStream:
    """
    Minimal YOLOv8-Pose + SORT streaming wrapper.

    - Uses Ultralytics YOLO-Pose for boxes + keypoints in one pass.
    - Tracks with SORT (IDs come from SORT).
    - Returns (1, num_person, 17, 3) per frame [x, y, conf], NaN-filled when absent.
    - Supports "slots" to pin specific people via a point/bbox prompt.
    """

    KEYPOINT_TO_IDX = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(
        self,
        model_name: str = "yolo11l-pose.pt",
        num_person: int = 1,
        *,
        device: Optional[str] = None,         # e.g., "cuda:0" or "cpu" (auto if None)
        pose_conf: float = 0.25,              # Ultralytics conf threshold
        iou_match_thresh: float = 0.3,        # IoU threshold to match slot->pose det
        sort_include_score: bool = True,      # pass score column to SORT output
        use_cache: bool = False,
        cache_dir: str = ".pose_cache",
    ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_model = YOLO(model_name)
        self.tracker = Sort(min_hits=0)

        self.num_person = int(num_person)
        self.pose_conf = float(pose_conf)
        self.iou_match_thresh = float(iou_match_thresh)
        self.sort_include_score = bool(sort_include_score)

        # --- NEW: cache setup ---
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._det_cache_mem: dict[str, dict] = {}

        # Stream state
        self._frames: List[np.ndarray] = []
        self._last_tracks: Optional[np.ndarray] = None

        # Slots for binding tracked people
        self._slots = [
            {"track_id": None, "label": None, "anchor": None, "anchor_bbox": None, "last_bbox": None}
            for _ in range(self.num_person)
        ]
        self._pending_prompts: List[Tuple[Tuple[float, float], Optional[str]]] = []

    # -------------------- Helpers --------------------

    def _hash_frame(self, frame: np.ndarray) -> str:
        """
        Hash the frame content + key inference settings so cache is safe
        across different models and thresholds.
        """
        # Ensure C-contiguous to keep .tobytes() stable
        f = np.ascontiguousarray(frame)
        h = hashlib.sha256()
        h.update(f.shape.__repr__().encode("utf-8"))
        h.update(f.dtype.str.encode("utf-8"))
        h.update(f.tobytes())
        # Include knobs that affect detections
        h.update(str(self.pose_conf).encode("utf-8"))
        return h.hexdigest()

    def _det_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _load_dets_from_cache(self, key: str) -> Optional[dict]:
        if not self.use_cache:
            return None
        if key in self._det_cache_mem:
            return self._det_cache_mem[key]
        p = self._det_cache_path(key)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                # light sanity check
                if isinstance(data, dict) and "det_boxes" in data and "kp_xy" in data:
                    self._det_cache_mem[key] = data
                    return data
            except Exception:
                pass
        return None

    def _save_dets_to_cache(self, key: str, dets: dict) -> None:
        if not self.use_cache:
            return
        self._det_cache_mem[key] = dets
        try:
            with open(self._det_cache_path(key), "wb") as f:
                pickle.dump(dets, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # best-effort; ignore disk write errors
            pass

    def clear_cache(self) -> None:
        """Clear in-memory detection cache (does not delete disk files)."""
        self._det_cache_mem.clear()


    @staticmethod
    def _prepare_dets_for_sort(boxes_xyxy: np.ndarray, scores: Optional[np.ndarray]) -> np.ndarray:
        """
        Build (N,5) = [x1,y1,x2,y2,score] for SORT.
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.empty((0, 5), dtype=np.float32)
        boxes_xyxy = boxes_xyxy.astype(np.float32)
        if scores is None:
            scores = np.ones((boxes_xyxy.shape[0],), dtype=np.float32)
        else:
            scores = scores.astype(np.float32).reshape(-1)
        return np.column_stack([boxes_xyxy, scores])

    @staticmethod
    def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def _nearest_track(self, center_xy: Tuple[float, float], tracks: np.ndarray) -> Tuple[int, Tuple[float, float, float, float]]:
        x, y = center_xy
        boxes = tracks[:, :4]
        ids = tracks[:, -1].astype(int)
        ctrs = np.column_stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5))
        idx = int(np.argmin((ctrs[:, 0] - x) ** 2 + (ctrs[:, 1] - y) ** 2))
        return ids[idx], tuple(map(float, boxes[idx]))

    # -------------------- Slot / Prompt API --------------------

    @staticmethod
    def _bbox_from_kps(kp_xy_single: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        kp_xy_single: (K,2) for one detection
        Returns (x1,y1,x2,y2) or None if all are NaN/invalid.
        """
        if kp_xy_single is None or kp_xy_single.size == 0:
            return None
        xy = kp_xy_single.astype(np.float32)
        valid = np.isfinite(xy).all(axis=1)
        if not np.any(valid):
            return None
        vxy = xy[valid]
        x1, y1 = float(np.min(vxy[:, 0])), float(np.min(vxy[:, 1]))
        x2, y2 = float(np.max(vxy[:, 0])), float(np.max(vxy[:, 1]))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def add_new_person_to_track(
        self,
        *,
        bbox: Tuple[float, float, float, float],
        label: Optional[str] = None,
    ) -> int:
        """
        Bind a new person slot using an anchor bounding box (x1,y1,x2,y2).
        Later, poses will be matched by IoU(anchor_bbox, pose_bbox_from_keypoints).
        """
        if bbox is None or len(bbox) != 4:
            raise ValueError("Provide bbox=(x1,y1,x2,y2).")

        free_slots = [i for i, s in enumerate(self._slots) if s.get("track_id") is None and s.get("anchor_bbox") is None]

        if not free_slots:
            raise ValueError("All slots are bound. Increase num_person or clear a slot.")

        slot_idx = free_slots[0]
        # Store the anchor bbox; we do NOT need _last_tracks for this new policy.
        self._slots[slot_idx].update(
            {"track_id": None, "label": label, "anchor": None, "anchor_bbox": tuple(map(float, bbox)), "last_bbox": None}
        )
        return slot_idx

    def _bind_pending_prompts(self, tracks: np.ndarray) -> None:
        if not self._pending_prompts or tracks is None or not tracks.size:
            return
        for center, label in list(self._pending_prompts):
            free = [i for i, s in enumerate(self._slots) if s["track_id"] is None]
            if not free:
                break
            slot_idx = free[0]
            track_id, track_bbox = self._nearest_track(center, tracks)
            self._slots[slot_idx].update(
                {"track_id": int(track_id), "label": label, "anchor": center, "last_bbox": track_bbox}
            )
            self._pending_prompts.pop(0)

    # -------------------- Core --------------------

    @torch.no_grad()
    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Returns array of shape (1, num_person, 17, 3): [x, y, conf].
        """
        K_TARGET = 17
        out = np.full((1, self.num_person, K_TARGET, 3), np.nan, dtype=np.float32)


        # 1) Ultralytics Pose on full frame (CACHED)
        # NOTE: even though the variable is named frame_rgb, if you are passing
        # cv2 frames, they are typically BGR. That's fine for caching because
        # hashing is on raw bytes; just be consistent across calls.
        det_boxes = det_scores = kp_xy = kp_conf = None

        key = self._hash_frame(frame_rgb)
        cached = self._load_dets_from_cache(key)
        if cached is not None:
            det_boxes = cached["det_boxes"]
            det_scores = cached["det_scores"]
            kp_xy = cached["kp_xy"]
            kp_conf = cached["kp_conf"]
        else:
            device_str = str(self.device) if self.device.type == "cuda" else "cpu"
            results = self.pose_model.predict(frame_rgb, conf=self.pose_conf, verbose=False, device=device_str)
            if not results:
                self._frames.append(out)
                return out

            res = results[0]
            if res.boxes is None or res.keypoints is None or res.boxes.xyxy is None:
                self._frames.append(out)
                return out

            # Extract detections
            det_boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N,4)
            det_scores = (
                res.boxes.conf.cpu().numpy().astype(np.float32)
                if getattr(res.boxes, "conf", None) is not None
                else np.ones((det_boxes.shape[0],), dtype=np.float32)
            )
            kp_xy = res.keypoints.xy.cpu().numpy().astype(np.float32)  # (N,K,2)
            kp_conf = (
                res.keypoints.conf.cpu().numpy().astype(np.float32)
                if getattr(res.keypoints, "conf", None) is not None
                else np.ones((kp_xy.shape[0], kp_xy.shape[1]), dtype=np.float32)
            )

            # Save to cache
            self._save_dets_to_cache(
                key,
                {
                    "det_boxes": det_boxes,
                    "det_scores": det_scores,
                    "kp_xy": kp_xy,
                    "kp_conf": kp_conf,
                },
            )

        # 2) Track with SORT using detection boxes
        dets_for_sort = self._prepare_dets_for_sort(det_boxes, det_scores)       # (N,5)
        tracks = self.tracker.update(dets_for_sort, include_score=self.sort_include_score)  # Nx(5 or 6)
        self._last_tracks = tracks.copy() if tracks is not None and tracks.size else None
        if tracks is None or tracks.shape[0] == 0:
            self._frames.append(out)
            return out

        # 3) Optionally bind queued prompts to current tracks
        self._bind_pending_prompts(tracks)

        # (rest of your method unchanged) ...
        # Build dictionary: track_id -> bbox
        last_col = tracks.shape[1] - 1
        id_to_box = {int(t[last_col]): tuple(map(float, t[:4])) for t in tracks}
        used_ids: set[int] = set()
        selected_boxes: List[Tuple[float, float, float, float]] = []

        # Fill with bound slots first
        for s in self._slots:
            box = None
            if s["track_id"] is not None and s["track_id"] in id_to_box:
                box = id_to_box[s["track_id"]]
                s["last_bbox"] = box
                used_ids.add(s["track_id"])
            elif s["last_bbox"] is not None:
                box = s["last_bbox"]
            if box is not None:
                selected_boxes.append(tuple(float(v) for v in box))
            if len(selected_boxes) >= self.num_person:
                break

        # Fill remaining with any other tracks
        if len(selected_boxes) < self.num_person:
            for tid, box in id_to_box.items():
                if tid in used_ids:
                    continue
                selected_boxes.append(tuple(float(v) for v in box))
                used_ids.add(tid)
                if len(selected_boxes) >= self.num_person:
                    break

        if not selected_boxes:
            self._frames.append(out)
            return out

        m_eff = min(len(selected_boxes), self.num_person)
        selected_boxes = selected_boxes[:m_eff]

        # Ensure K=17 (slice or pad)
        K_model = kp_xy.shape[1]
        if K_model != K_TARGET:
            if K_model > K_TARGET:
                kp_xy = kp_xy[:, :K_TARGET, :]
                kp_conf = kp_conf[:, :K_TARGET]
            else:
                xy_pad = np.full((kp_xy.shape[0], K_TARGET, 2), np.nan, dtype=np.float32)
                c_pad = np.zeros((kp_conf.shape[0], K_TARGET), dtype=np.float32)
                xy_pad[:, :K_model, :] = kp_xy
                c_pad[:, :K_model] = kp_conf
                kp_xy, kp_conf = xy_pad, c_pad

        # Build per-pose keypoint-derived bboxes
        pose_boxes: List[Optional[Tuple[float, float, float, float]]] = [
            self._bbox_from_kps(kp_xy[j]) for j in range(kp_xy.shape[0])
        ]

        # 4) Match each selected box to best pose det via IoU
        used_pose_idx: set[int] = set()

        # --- Priority 1: Fill slots that have an anchor_bbox via IoU(anchor_bbox, pose_bbox) ---
        for i, s in enumerate(self._slots[:self.num_person]):
            if s.get("anchor_bbox") is None:
                continue
            anchor_box = s["anchor_bbox"]
            best_idx, best_iou = -1, 0.0
            for j, pbox in enumerate(pose_boxes):
                if j in used_pose_idx or pbox is None:
                    continue
                iou = self._iou(anchor_box, pbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, j

            if best_idx >= 0 and best_iou >= self.iou_match_thresh:
                out[0, i, :, :2] = kp_xy[best_idx]
                out[0, i, :, 2]  = kp_conf[best_idx]
                used_pose_idx.add(best_idx)
                # keep last_bbox for visualization; prefer the keypoint box for stability here
                s["last_bbox"] = pose_boxes[best_idx]

        # --- Priority 2: Fill any remaining slots using previous SORT-driven selection as fallback ---
        # Build dictionary: track_id -> bbox
        last_col = tracks.shape[1] - 1
        id_to_box = {int(t[last_col]): tuple(map(float, t[:4])) for t in tracks} if tracks is not None and tracks.size else {}
        used_ids: set[int] = set()
        selected_boxes: List[Tuple[float, float, float, float]] = []

        # Reuse previously bound slots (track-based) if they weren't filled via anchor bbox
        for i, s in enumerate(self._slots[:self.num_person]):
            if not np.isnan(out[0, i, :, 2]).all():  # already filled by anchor bbox stage
                continue
            box = None
            if s["track_id"] is not None and s["track_id"] in id_to_box:
                box = id_to_box[s["track_id"]]
                s["last_bbox"] = box
                used_ids.add(s["track_id"])
            elif s["last_bbox"] is not None:
                box = s["last_bbox"]
            if box is not None:
                selected_boxes.append(tuple(float(v) for v in box))

        # Fill remaining with any other tracks
        if len(selected_boxes) < self.num_person:
            for tid, box in id_to_box.items():
                if tid in used_ids:
                    continue
                selected_boxes.append(tuple(float(v) for v in box))
                used_ids.add(tid)
                if len(selected_boxes) >= self.num_person:
                    break

        # Now map those selected (track) boxes to any remaining unused poses via IoU
        pose_needed_slots = [i for i in range(self.num_person) if np.isnan(out[0, i, :, 2]).all()]
        for i_rel, sel_box in enumerate(selected_boxes[:len(pose_needed_slots)]):
            slot_i = pose_needed_slots[i_rel]
            best_idx, best_iou = -1, 0.0
            for j, pbox in enumerate(pose_boxes):
                if j in used_pose_idx or pbox is None:
                    continue
                iou = self._iou(sel_box, pbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_idx >= 0 and best_iou >= self.iou_match_thresh:
                out[0, slot_i, :, :2] = kp_xy[best_idx]
                out[0, slot_i, :, 2]  = kp_conf[best_idx]
                used_pose_idx.add(best_idx)
                # track a last_bbox for downstream visualization
                self._slots[slot_i]["last_bbox"] = sel_box


        # Update slot last bboxes for displayed slots
        for i, s in enumerate(self._slots[:m_eff]):
            s["last_bbox"] = selected_boxes[i]

        self._frames.append(out)
        return out

    # -------------------- Video I/O & results --------------------

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

    def reset_results(self) -> None:
        self._frames.clear()

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

    # -------------------- Optional helpers --------------------

    def get_slot_labels(self) -> List[Optional[str]]:
        return [s["label"] for s in self._slots]

    def clear_slot(self, slot_idx: int) -> None:
        self._slots[slot_idx] = {
            "track_id": None, "label": None, "anchor": None, "anchor_bbox": None, "last_bbox": None
        }

    def slots_info(self) -> List[Dict]:
        return copy.deepcopy(self._slots)

    def clear(self, *, keep_slot_labels: bool = False, keep_pending_prompts: bool = False) -> None:
        """
        Reset streamer state for a new video.
        """
        self.reset_results()
        self._last_tracks = None
        self.tracker = Sort(min_hits=0)

        if keep_slot_labels:
            for s in self._slots:
                s["track_id"] = None
                s["anchor"] = None
                s["anchor_bbox"] = None
                s["last_bbox"] = None
        else:
            for i in range(len(self._slots)):
                self._slots[i] = {
                    "track_id": None, "label": None, "anchor": None, "anchor_bbox": None, "last_bbox": None
                }

        if not keep_pending_prompts:
            self._pending_prompts.clear()


if __name__ == "__main__":
    video_path = orig_video_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/S00047/S00047_brushing3_1.mkv"

    streamer = Pose2DStream(model_name="yolo11x-pose.pt", num_person=1, device="cuda:0")

    import cv2

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    streamer.process_frame(frame)