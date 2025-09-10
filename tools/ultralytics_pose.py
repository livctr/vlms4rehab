from __future__ import annotations

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

    def __init__(
        self,
        model_name: str = "yolo11l-pose.pt",
        num_person: int = 1,
        *,
        device: Optional[str] = None,         # e.g., "cuda:0" or "cpu" (auto if None)
        pose_conf: float = 0.25,              # Ultralytics conf threshold
        iou_match_thresh: float = 0.3,        # IoU threshold to match slot->pose det
        sort_include_score: bool = True,      # pass score column to SORT output
    ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_model = YOLO(model_name)
        self.tracker = Sort(min_hits=0)

        self.num_person = int(num_person)
        self.pose_conf = float(pose_conf)
        self.iou_match_thresh = float(iou_match_thresh)
        self.sort_include_score = bool(sort_include_score)

        # Stream state
        self._frames: List[np.ndarray] = []
        self._last_tracks: Optional[np.ndarray] = None

        # Slots for binding tracked people
        self._slots: List[Dict] = [
            {"track_id": None, "label": None, "anchor": None, "last_bbox": None}
            for _ in range(self.num_person)
        ]
        self._pending_prompts: List[Tuple[Tuple[float, float], Optional[str]]] = []

    # -------------------- Helpers --------------------

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

    def add_new_person_to_track(
        self,
        point: Optional[Tuple[float, float]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        label: Optional[str] = None,
    ) -> int:
        if (point is None) == (bbox is None):
            raise ValueError("Provide exactly one of: point=(x,y) OR bbox=(x1,y1,x2,y2).")

        center = point if point is not None else ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

        free_slots = [i for i, s in enumerate(self._slots) if s["track_id"] is None and s["anchor"] is None]
        if not free_slots:
            raise ValueError("All slots are bound. Increase num_person or clear a slot.")

        slot_idx = free_slots[0]
        if self._last_tracks is not None and self._last_tracks.size:
            track_id, track_bbox = self._nearest_track(center, self._last_tracks)
            self._slots[slot_idx].update(
                {"track_id": int(track_id), "label": label, "anchor": center, "last_bbox": track_bbox}
            )
        else:
            self._slots[slot_idx].update({"track_id": None, "label": label, "anchor": center, "last_bbox": None})
            self._pending_prompts.append((center, label))
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
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns array of shape (1, num_person, 17, 3): [x, y, conf].
        """
        K_TARGET = 17
        out = np.full((1, self.num_person, K_TARGET, 3), np.nan, dtype=np.float32)

        # 1) Ultralytics Pose on full frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        det_boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)              # (N,4)
        det_scores = (res.boxes.conf.cpu().numpy().astype(np.float32)
                      if getattr(res.boxes, "conf", None) is not None
                      else np.ones((det_boxes.shape[0],), dtype=np.float32))     # (N,)

        kp_xy = res.keypoints.xy.cpu().numpy().astype(np.float32)               # (N,K,2)
        kp_conf = (res.keypoints.conf.cpu().numpy().astype(np.float32)
                   if getattr(res.keypoints, "conf", None) is not None
                   else np.ones((kp_xy.shape[0], kp_xy.shape[1]), dtype=np.float32))  # (N,K)

        # 2) Track with SORT using detection boxes
        dets_for_sort = self._prepare_dets_for_sort(det_boxes, det_scores)       # (N,5)
        tracks = self.tracker.update(dets_for_sort, include_score=self.sort_include_score)  # Nx(5 or 6)
        self._last_tracks = tracks.copy() if tracks is not None and tracks.size else None
        if tracks is None or tracks.shape[0] == 0:
            self._frames.append(out)
            return out

        # 3) Optionally bind queued prompts to current tracks
        self._bind_pending_prompts(tracks)

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

        # 4) Match each selected box to best pose det via IoU
        used_pose_idx: set[int] = set()
        for i, sel_box in enumerate(selected_boxes):
            best_idx, best_iou = -1, 0.0
            for j, det_box in enumerate(det_boxes):
                if j in used_pose_idx:
                    continue
                iou = self._iou(sel_box, tuple(map(float, det_box)))
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_idx >= 0 and best_iou >= self.iou_match_thresh:
                out[0, i, :, :2] = kp_xy[best_idx]
                out[0, i, :, 2] = kp_conf[best_idx]
                used_pose_idx.add(best_idx)

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
        self._slots[slot_idx] = {"track_id": None, "label": None, "anchor": None, "last_bbox": None}

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
                s["last_bbox"] = None
        else:
            for i in range(len(self._slots)):
                self._slots[i] = {"track_id": None, "label": None, "anchor": None, "last_bbox": None}

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