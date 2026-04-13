from typing import Dict, List, Optional, Tuple

import numpy as np


class Sort:
    """Lightweight SORT-compatible tracker used by Pose2DStream.

    This is an IoU-based ID association fallback that preserves a stable
    `update(dets, include_score=...)` API expected by existing code.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_hits: int = 0,
        max_age: int = 1,
    ):
        self.iou_threshold = float(iou_threshold)
        self.min_hits = int(min_hits)
        self.max_age = int(max_age)

        self._next_id = 1
        self._tracks: Dict[int, Dict[str, object]] = {}

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    def _match_track_id(self, box: np.ndarray, unmatched_ids: List[int]) -> Optional[int]:
        best_id = None
        best_iou = 0.0
        for tid in unmatched_ids:
            prev_box = self._tracks[tid]["bbox"]
            iou = self._iou(box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        if best_id is not None and best_iou >= self.iou_threshold:
            return int(best_id)
        return None

    def update(self, dets: np.ndarray, include_score: bool = True) -> np.ndarray:
        """Track detections and return rows ending with track id.

        Args:
            dets: shape (N,4) or (N,5) where columns are [x1,y1,x2,y2,(score)].
            include_score: when True returns [x1,y1,x2,y2,score,id],
                else [x1,y1,x2,y2,id].
        """
        if dets is None or dets.size == 0:
            stale_ids = []
            for tid, t in self._tracks.items():
                t["age"] = int(t["age"]) + 1
                if int(t["age"]) > self.max_age:
                    stale_ids.append(tid)
            for tid in stale_ids:
                self._tracks.pop(tid, None)
            return np.empty((0, 6 if include_score else 5), dtype=np.float32)

        if dets.ndim != 2 or dets.shape[1] < 4:
            raise ValueError("dets must have shape (N,4) or (N,5)")

        boxes = dets[:, :4].astype(np.float32)
        if dets.shape[1] >= 5:
            scores = dets[:, 4].astype(np.float32)
        else:
            scores = np.ones((dets.shape[0],), dtype=np.float32)

        unmatched_ids = list(self._tracks.keys())
        assigned_ids: List[int] = []

        for i, box in enumerate(boxes):
            tid = self._match_track_id(box, unmatched_ids)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            else:
                unmatched_ids.remove(tid)

            self._tracks[tid] = {
                "bbox": box.copy(),
                "score": float(scores[i]),
                "age": 0,
            }
            assigned_ids.append(int(tid))

        stale_ids = []
        for tid in unmatched_ids:
            self._tracks[tid]["age"] = int(self._tracks[tid]["age"]) + 1
            if int(self._tracks[tid]["age"]) > self.max_age:
                stale_ids.append(tid)
        for tid in stale_ids:
            self._tracks.pop(tid, None)

        if include_score:
            return np.column_stack([boxes, scores, np.asarray(assigned_ids, dtype=np.float32)])
        return np.column_stack([boxes, np.asarray(assigned_ids, dtype=np.float32)])
