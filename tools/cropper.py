from __future__ import annotations
from typing import Optional, Tuple, List, Any, Dict, Union
import numpy as np
from tools.ultralytics_pose import Pose2DStream
from tools.vqa.qwen2_5_vl import Qwen2_5_VL_VQA
import json

Number = Union[int, float]

def _to_float4(b: Union[List[Number], Tuple[Number, Number, Number, Number]]) -> Tuple[float, float, float, float]:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError(f"Expected bbox of length 4, got {type(b).__name__} with len={len(b) if hasattr(b, '__len__') else 'N/A'}")
    x1, y1, x2, y2 = b
    for v in (x1, y1, x2, y2):
        if not isinstance(v, (int, float)):
            raise TypeError(f"BBox elements must be numbers, got {type(v).__name__}")
    return float(x1), float(y1), float(x2), float(y2)

def _strip_to_json_array(s: str) -> str:
    s = s.strip()
    start = s.find('[')
    end = s.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a JSON array in the input string.")
    return s[start:end+1]

def extract_first_bbox_and_label(
    detections: Union[str, List[Dict[str, Any]]]
) -> Tuple[Tuple[float, float, float, float], Optional[str]]:
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

    raw_bbox = first.get("bbox_2d", first.get("bbox"))
    if raw_bbox is None:
        raise KeyError("Detection missing 'bbox_2d' (or 'bbox') key.")

    bbox = _to_float4(raw_bbox)
    label = first.get("label")
    return bbox, label






from PIL import Image, ImageDraw

def draw_hand_debug_on_frame(frame_rgb: np.ndarray,
                             kp_17x3: np.ndarray,
                             handedness: str,
                             center_xy: Tuple[float, float],
                             r: int = 5) -> Image.Image:
    """
    frame_rgb: (H, W, 3) uint8 RGB
    kp_17x3:  (17, 3) [x, y, conf]
    """
    img = Image.fromarray(frame_rgb)  # assumes RGB, no conversion
    draw = ImageDraw.Draw(img)

    # elbow / wrist indices
    if handedness.lower() == "right":
        elbow_i, wrist_i = 8, 10
    else:
        elbow_i, wrist_i = 7, 9

    # extract and guard
    def pt(i):
        x, y, c = kp_17x3[i]
        return (float(x), float(y), float(c))

    ex, ey, ec = pt(elbow_i)
    wx, wy, wc = pt(wrist_i)
    cx, cy = center_xy

    # elbow (blue)
    if np.isfinite(ex) and np.isfinite(ey) and ec > 0:
        draw.ellipse((ex - r, ey - r, ex + r, ey + r), outline=(0, 102, 255), width=2)
        draw.line((ex, ey, wx, wy), fill=(0, 102, 255), width=2) if np.isfinite(wx) and np.isfinite(wy) else None

    # wrist (red)
    if np.isfinite(wx) and np.isfinite(wy) and wc > 0:
        draw.ellipse((wx - r, wy - r, wx + r, wy + r), outline=(255, 0, 0), width=2)

    # center (green)
    if np.isfinite(cx) and np.isfinite(cy):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 200, 0), width=3)

    return img



class HandCropper:
    def __init__(
        self,
        stream: Optional[Pose2DStream],
        vqa_model: Optional[Qwen2_5_VL_VQA],
        *,
        kp_conf_thresh: float = 0.91,
    ):
        self.kp_conf_thresh = float(kp_conf_thresh)

        self.stream = stream
        self.vqa_model = vqa_model

        self.side = 224
        self.half = self.side // 2

        self._last_center: Optional[Tuple[float, float]] = None
        self._detected = False
    
    def clear(self) -> None:
        # flush tracker + slots + pending prompts
        self.stream.clear(keep_slot_labels=False, keep_pending_prompts=False)
        # reset cropper state
        self._last_center = None
        self._detected = False
        # optional: if your VQA has no clear(), guard it
        if hasattr(self.vqa_model, "clear") and callable(self.vqa_model.clear):
            self.vqa_model.clear()

    def _handedness_to_idx(self, handedness: str) -> Tuple[int, int]:
        handedness = handedness.lower()
        assert handedness in ("left", "right")
        K2I = self.stream.KEYPOINT_TO_IDX
        return (K2I["right_elbow"], K2I["right_wrist"]) if handedness == "right" else (K2I["left_elbow"], K2I["left_wrist"])

    def _get_kps(self, frame: np.ndarray) -> np.ndarray:
        """Return (17,3) keypoints for the first/selected person on a single frame."""
        kps = self.stream.process_frame(frame)  # (1, num_person, 17, 3)
        return kps[0, 0]

    def _clamp_center(self, cx: float, cy: float, W: int, H: int) -> Tuple[int, int]:
        cx = max(self.half, min(W - self.half, cx))
        cy = max(self.half, min(H - self.half, cy))
        return int(round(cx)), int(round(cy))

    def _center_from_kps(self, kp: np.ndarray, handedness: str) -> Tuple[Tuple[float, float], float, float]:
        """
        Compute center from a (17,3) kp array using the 5:1 elbow->wrist rule.
        Returns: (center_xy), wrist_conf, elbow_conf
        """
        kp_elbow, kp_wrist = self._handedness_to_idx(handedness)
        wx, wy, wc = kp[kp_wrist]
        ex, ey, ec = kp[kp_elbow]
        # center = wrist + (elbow->wrist)/5
        vx, vy = (wx - ex), (wy - ey)
        cx, cy = (wx + vx / 5.0), (wy + vy / 5.0)
        return (float(cx), float(cy)), float(wc), float(ec)

    def _center_from_frame(self, frame: np.ndarray, handedness: str) -> Tuple[Tuple[float, float], float, float]:
        """
        Run pose on a single frame and compute center + confidences.
        Returns: (center_xy), wrist_conf, elbow_conf
        """
        kp = self._get_kps(frame)
        return self._center_from_kps(kp, handedness)

    def _ensure_detection(self, frame: np.ndarray) -> None:
        if self._detected:
            return
        if self.vqa_model is None:
            self._detected = True
            return
        prompt = "Locate the patient as a bounding box in JSON. If there are multiple people, choose the one closer to the camera."
        patient_loc_text = self.vqa_model.process_frames(frame, context=prompt)
        bbox, label = extract_first_bbox_and_label(patient_loc_text)
        self.stream.add_new_person_to_track(bbox=bbox, label=label)
        self._detected = True

    @staticmethod
    def _bbox_contains_points(
        bbox: Tuple[int, int, int, int],
        points: Tuple[Tuple[float, float], ...]
    ) -> bool:
        x1, y1, x2, y2 = bbox
        for (px, py) in points:
            if not (x1 <= px <= x2 and y1 <= py <= y2):
                return False
        return True

    def _bbox_at_center_with_side(
        self, center: Tuple[float, float], side: int, W: int, H: int
    ) -> Tuple[int, int, int, int]:
        half = side // 2
        cx_i, cy_i = self._clamp_center(center[0], center[1], W, H)
        x1, y1 = int(cx_i - half), int(cy_i - half)
        x2, y2 = int(cx_i + half), int(cy_i + half)
        # clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        return x1, y1, x2, y2

    def process_frames(self, frames: np.ndarray, *, handedness: str) -> List[Tuple[int,int,int,int]]:
        """
        Return a list of per-frame bounding boxes (x1, y1, x2, y2), ints.

        If both wrist & elbow at the first AND last frames meet kp_conf_thresh:
        - sophisticated=True  -> linearly interpolate centers between first & last; box size = self.side
        - sophisticated=False -> use first-frame center for all frames
        Else: return full-frame boxes for all frames.
        """
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        # Do NOT clear slots/tracker here; caller calls .clear() once per video.
        # Only reset per-call keypoint RESULT buffers (safe), not slots.
        self.stream.reset_results()

        # Ensure detection at most once per video
        self._ensure_detection(frames[0])

        # Helper: bbox (fixed square of size self.side) centered at (cx, cy), clamped to image bounds
        # --- Endpoint check (gate by kp_conf_thresh) ---
        c0, wc0, ec0 = self._center_from_frame(frames[0], handedness)
        c1, wc1, ec1 = self._center_from_frame(frames[-1], handedness)
        first_ok = (wc0 >= self.kp_conf_thresh) and (ec0 >= self.kp_conf_thresh)
        last_ok  = (wc1 >= self.kp_conf_thresh) and (ec1 >= self.kp_conf_thresh)

        if not (first_ok and last_ok):
            # Not confident enough at the endpoints -> full-frame boxes
            full = (0, 0, int(W), int(H))
            return [full for _ in range(T)]

        # --- Confident: produce bboxes according to mode ---
        # Use the center between the start and end frames
        # Compute midpoint center
        cmid = ((c0[0] + c1[0]) / 2.0, (c0[1] + c1[1]) / 2.0)

        margin = 168
        crops = [224, 448, 672]

        for crop in crops:
            inner = self._bbox_at_center_with_side(cmid, side=crop-margin, W=W, H=H)
            if self._bbox_contains_points(inner, (c0, c1)):
                outer = self._bbox_at_center_with_side(cmid, side=crop, W=W, H=H)
                return [outer for _ in range(T)]
        return [(0, 0, int(W), int(H)) for _ in range(T)]



class HandPointer:
    def __init__(
        self,
        stream: Optional[Pose2DStream],
        vqa_model: Optional[Qwen2_5_VL_VQA],
        *,
        kp_conf_thresh: float = 0.91,
        fast_movement_thresh: int = 56,
        crop_size: int = 224,
    ):
        self.kp_conf_thresh = float(kp_conf_thresh)
        self.fast_movement_thresh = int(fast_movement_thresh)
        self.side = int(crop_size)
        self.half = self.side // 2

        self.stream = stream
        self.vqa_model = vqa_model

        self._last_center: Optional[Tuple[float, float]] = None
        self._detected = False
    
    def clear(self) -> None:
        # flush tracker + slots + pending prompts
        self.stream.clear(keep_slot_labels=False, keep_pending_prompts=False)
        # reset cropper state
        self._last_center = None
        self._detected = False
        # optional: if your VQA has no clear(), guard it
        if hasattr(self.vqa_model, "clear") and callable(self.vqa_model.clear):
            self.vqa_model.clear()

    def _handedness_to_idx(self, handedness: str) -> Tuple[int, int]:
        handedness = handedness.lower()
        assert handedness in ("left", "right")
        K2I = self.stream.KEYPOINT_TO_IDX
        return (K2I["right_elbow"], K2I["right_wrist"]) if handedness == "right" else (K2I["left_elbow"], K2I["left_wrist"])

    def _get_kps(self, frame: np.ndarray) -> np.ndarray:
        """Return (17,3) keypoints for the first/selected person on a single frame."""
        kps = self.stream.process_frame(frame)  # (1, num_person, 17, 3)
        return kps[0, 0]

    def _clamp_center(self, cx: float, cy: float, W: int, H: int) -> Tuple[int, int]:
        cx = max(self.half, min(W - self.half, cx))
        cy = max(self.half, min(H - self.half, cy))
        return int(round(cx)), int(round(cy))

    def _center_from_kps(self, kp: np.ndarray, handedness: str) -> Tuple[Tuple[float, float], float, float]:
        """
        Compute center from a (17,3) kp array using the 5:1 elbow->wrist rule.
        Returns: (center_xy), wrist_conf, elbow_conf
        """
        kp_elbow, kp_wrist = self._handedness_to_idx(handedness)
        wx, wy, wc = kp[kp_wrist]
        ex, ey, ec = kp[kp_elbow]
        # center = wrist + (elbow->wrist)/5
        vx, vy = (wx - ex), (wy - ey)
        cx, cy = (wx + vx / 5.0), (wy + vy / 5.0)
        return (float(cx), float(cy)), float(wc), float(ec)

    def _center_from_frame(self, frame: np.ndarray, handedness: str) -> Tuple[Tuple[float, float], float, float]:
        """
        Run pose on a single frame and compute center + confidences.
        Returns: (center_xy), wrist_conf, elbow_conf
        """
        kp = self._get_kps(frame)
        return self._center_from_kps(kp, handedness)

    def _ensure_detection(self, frame: np.ndarray) -> None:
        if self._detected:
            return
        if self.vqa_model is None:
            self._detected = True
            return
        prompt = "Locate the patient as a bounding box in JSON. If there are multiple people, choose the one closer to the camera."
        patient_loc_text = self.vqa_model.process_frames(frame, context=prompt)
        bbox, label = extract_first_bbox_and_label(patient_loc_text)
        self.stream.add_new_person_to_track(bbox=bbox, label=label)
        self._detected = True

    @staticmethod
    def _bbox_contains_points(
        bbox: Tuple[int, int, int, int],
        points: Tuple[Tuple[float, float], ...]
    ) -> bool:
        x1, y1, x2, y2 = bbox
        for (px, py) in points:
            if not (x1 <= px <= x2 and y1 <= py <= y2):
                return False
        return True

    def _bbox_at_center_with_side(
        self, center: Tuple[float, float], side: int, W: int, H: int
    ) -> Tuple[int, int, int, int]:
        half = side // 2
        cx_i, cy_i = self._clamp_center(center[0], center[1], W, H)
        x1, y1 = int(cx_i - half), int(cy_i - half)
        x2, y2 = int(cx_i + half), int(cy_i + half)
        # clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        return x1, y1, x2, y2

    def process_frames(self, frames: np.ndarray, *, handedness: str, interpolation = "none") -> List[Tuple[int,int,int,int]]:
        """
        Return a list of per-frame bounding boxes (x1, y1, x2, y2), ints.

        If both wrist & elbow at the first AND last frames meet kp_conf_thresh:
        - sophisticated=True  -> linearly interpolate centers between first & last; box size = self.side
        - sophisticated=False -> use first-frame center for all frames
        Else: return full-frame boxes for all frames.
        """
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        # Do NOT clear slots/tracker here; caller calls .clear() once per video.
        # Only reset per-call keypoint RESULT buffers (safe), not slots.
        self.stream.reset_results()

        # Ensure detection at most once per video
        self._ensure_detection(frames[0])

        # Helper: bbox (fixed square of size self.side) centered at (cx, cy), clamped to image bounds
        # --- Endpoint check (gate by kp_conf_thresh) ---
        c0, wc0, ec0 = self._center_from_frame(frames[0], handedness)
        c1, wc1, ec1 = self._center_from_frame(frames[-1], handedness)
        first_ok = (wc0 >= self.kp_conf_thresh) and (ec0 >= self.kp_conf_thresh)
        last_ok  = (wc1 >= self.kp_conf_thresh) and (ec1 >= self.kp_conf_thresh)

        if not (first_ok and last_ok):
            # Not confident enough at the endpoints -> full-frame boxes
            full = (0, 0, int(W), int(H))
            return "ABSTAIN", [full for _ in range(T)]
        
        # --- Confident: produce bboxes according to mode ---
        # Use the center between the start and end frames
        # Compute midpoint center
        dist = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))
        cmid = ((c0[0] + c1[0]) / 2.0, (c0[1] + c1[1]) / 2.0)
        cmid = self._clamp_center(cmid[0], cmid[1], W, H)

        if interpolation == "none":
            crop_box = self._bbox_at_center_with_side(cmid, side=self.side, W=W, H=H)
            if dist >= self.fast_movement_thresh:
                return f"FAST MOVEMENT [{dist:<.2f}]", [crop_box for _ in range(T)]
            return f"OK [{dist:<.2f}]", [crop_box for _ in range(T)]
        elif interpolation == "linear":
            c0box = self._bbox_at_center_with_side(c0, side=self.side, W=W, H=H)
            c1box = self._bbox_at_center_with_side(c1, side=self.side, W=W, H=H)
            boxes = []
            for i in range(T):
                alpha = i / (T - 1) if T > 1 else 0.0
                x1 = int(round((1 - alpha) * c0box[0] + alpha * c1box[0]))
                y1 = int(round((1 - alpha) * c0box[1] + alpha * c1box[1]))
                x2 = int(round((1 - alpha) * c0box[2] + alpha * c1box[2]))
                y2 = int(round((1 - alpha) * c0box[3] + alpha * c1box[3]))
                boxes.append((x1, y1, x2, y2))
            if dist >= self.fast_movement_thresh:
                return f"FAST MOVEMENT [{dist:<.2f}]", boxes
            return f"OK [{dist:<.2f}]", boxes
        elif interpolation == "individual":
            boxes = []
            for i in range(T):
                ci, wci, eci = self._center_from_frame(frames[i], handedness)
                if (wci >= self.kp_conf_thresh) and (eci >= self.kp_conf_thresh):
                    boxi = self._bbox_at_center_with_side(ci, side=self.side, W=W, H=H)
                else:
                    boxi = (0, 0, W, H)
                boxes.append(boxi)
            return "INDIVIDUAL", boxes
        else:
            raise ValueError(f"Unknown interpolation mode: {interpolation}")

class HandPointerV2:
    """Also outputs the wrist and elbow keypoints."""
    def __init__(
        self,
        stream: Optional[Pose2DStream],
        vqa_model: Optional[Qwen2_5_VL_VQA],
        *,
        kp_conf_thresh: float = 0.91,
        fast_movement_thresh: int = 56,
        crop_size: int = 224,
    ):
        self.kp_conf_thresh = float(kp_conf_thresh)
        self.fast_movement_thresh = int(fast_movement_thresh)
        self.side = int(crop_size)
        self.half = self.side // 2

        self.stream = stream
        self.vqa_model = vqa_model

        self._last_center: Optional[Tuple[float, float]] = None
        self._detected = False
    
    def clear(self) -> None:
        # flush tracker + slots + pending prompts
        self.stream.clear(keep_slot_labels=False, keep_pending_prompts=False)
        # reset cropper state
        self._last_center = None
        self._detected = False
        # optional: if your VQA has no clear(), guard it
        if hasattr(self.vqa_model, "clear") and callable(self.vqa_model.clear):
            self.vqa_model.clear()

    def _handedness_to_idx(self, handedness: str) -> Tuple[int, int]:
        handedness = handedness.lower()
        assert handedness in ("left", "right")
        K2I = self.stream.KEYPOINT_TO_IDX
        return (K2I["right_elbow"], K2I["right_wrist"]) if handedness == "right" else (K2I["left_elbow"], K2I["left_wrist"])

    def _get_kps(self, frame: np.ndarray) -> np.ndarray:
        """Return (17,3) keypoints for the first/selected person on a single frame."""
        kps = self.stream.process_frame(frame)  # (1, num_person, 17, 3)
        return kps[0, 0]

    def _clamp_center(self, cx: float, cy: float, W: int, H: int) -> Tuple[int, int]:
        cx = max(self.half, min(W - self.half, cx))
        cy = max(self.half, min(H - self.half, cy))
        return int(round(cx)), int(round(cy))

    def _center_from_kps(self, kp: np.ndarray, handedness: str) -> Tuple[Tuple[float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Compute center from a (17,3) kp array using the 5:1 elbow->wrist rule.
        Returns: (center_xy), wrist_conf, elbow_conf
        """
        kp_elbow, kp_wrist = self._handedness_to_idx(handedness)
        wx, wy, wc = kp[kp_wrist]
        ex, ey, ec = kp[kp_elbow]
        # center = wrist + (elbow->wrist)/5
        vx, vy = (wx - ex), (wy - ey)
        cx, cy = (wx + vx / 5.0), (wy + vy / 5.0)
        return (float(cx), float(cy)), kp[kp_wrist], kp[kp_elbow]

    def _center_from_frame(self, frame: np.ndarray, handedness: str) -> Tuple[Tuple[float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Run pose on a single frame and compute center + confidences.
        Returns: (center_xy), wrist_conf, elbow_conf
        """
        kp = self._get_kps(frame)
        return self._center_from_kps(kp, handedness)

    def _ensure_detection(self, frame: np.ndarray) -> None:
        if self._detected:
            return
        if self.vqa_model is None:
            self._detected = True
            return
        prompt = "Locate the patient as a bounding box in JSON. If there are multiple people, choose the one closer to the camera."
        patient_loc_text = self.vqa_model.process_frames(frame, context=prompt)
        bbox, label = extract_first_bbox_and_label(patient_loc_text)
        self.stream.add_new_person_to_track(bbox=bbox, label=label)
        self._detected = True

    @staticmethod
    def _bbox_contains_points(
        bbox: Tuple[int, int, int, int],
        points: Tuple[Tuple[float, float], ...]
    ) -> bool:
        x1, y1, x2, y2 = bbox
        for (px, py) in points:
            if not (x1 <= px <= x2 and y1 <= py <= y2):
                return False
        return True

    def _bbox_at_center_with_side(
        self, center: Tuple[float, float], side: int, W: int, H: int
    ) -> Tuple[int, int, int, int]:
        half = side // 2
        cx_i, cy_i = self._clamp_center(center[0], center[1], W, H)
        x1, y1 = int(cx_i - half), int(cy_i - half)
        x2, y2 = int(cx_i + half), int(cy_i + half)
        # clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        return x1, y1, x2, y2

    def process_frames(self, frames: np.ndarray, *, handedness: str, interpolation = "none") -> List[Tuple[int,int,int,int]]:
        """
        Return a list of per-frame bounding boxes (x1, y1, x2, y2), ints.

        If both wrist & elbow at the first AND last frames meet kp_conf_thresh:
        - sophisticated=True  -> linearly interpolate centers between first & last; box size = self.side
        - sophisticated=False -> use first-frame center for all frames
        Else: return full-frame boxes for all frames.
        """
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        # Do NOT clear slots/tracker here; caller calls .clear() once per video.
        # Only reset per-call keypoint RESULT buffers (safe), not slots.
        self.stream.reset_results()

        # Ensure detection at most once per video
        self._ensure_detection(frames[0])

        # Helper: bbox (fixed square of size self.side) centered at (cx, cy), clamped to image bounds
        # --- Endpoint check (gate by kp_conf_thresh) ---
        centers, wrist_kps, elbow_kps, oks = [], [], [], []
        for i in range(T):
            ci, wci, eci = self._center_from_frame(frames[i], handedness)
            ok = (wci[2] >= self.kp_conf_thresh) and (eci[2] >= self.kp_conf_thresh)
            centers.append(ci)
            wrist_kps.append(wci)
            elbow_kps.append(eci)
            oks.append(ok)
        
        if interpolation == "individual":
            boxes = []
            for i in range(T):
                if oks[i]:
                    boxi = self._bbox_at_center_with_side(centers[i], side=self.side, W=W, H=H)
                else:
                    boxi = (0, 0, W, H)
                boxes.append(boxi)
            return "INDIVIDUAL", boxes, wrist_kps, elbow_kps
        elif interpolation == "linear":
            c0box = self._bbox_at_center_with_side(centers[0], side=self.side, W=W, H=H)
            c1box = self._bbox_at_center_with_side(centers[-1], side=self.side, W=W, H=H)
            boxes = []
            for i in range(T):
                alpha = i / (T - 1) if T > 1 else 0.0
                x1 = int(round((1 - alpha) * c0box[0] + alpha * c1box[0]))
                y1 = int(round((1 - alpha) * c0box[1] + alpha * c1box[1]))
                x2 = int(round((1 - alpha) * c0box[2] + alpha * c1box[2]))
                y2 = int(round((1 - alpha) * c0box[3] + alpha * c1box[3]))
                boxes.append((x1, y1, x2, y2))
            dist = max(abs(centers[-1][0] - centers[0][0]), abs(centers[-1][1] - centers[0][1]))
            if dist >= self.fast_movement_thresh:
                return f"FAST MOVEMENT [{dist:<.2f}]", boxes, wrist_kps, elbow_kps
            return f"OK [{dist:<.2f}]", boxes, wrist_kps, elbow_kps
        elif interpolation == "none":
            c0box = self._bbox_at_center_with_side(centers[0], side=self.side, W=W, H=H)
            dist = max(abs(centers[-1][0] - centers[0][0]), abs(centers[-1][1] - centers[0][1]))
            if dist >= self.fast_movement_thresh:
                return f"FAST MOVEMENT [{dist:<.2f}]", [c0box for _ in range(T)], wrist_kps, elbow_kps
            return f"OK [{dist:<.2f}]", [c0box for _ in range(T)], wrist_kps, elbow_kps
        else:
            raise ValueError(f"Unknown interpolation mode: {interpolation}")
