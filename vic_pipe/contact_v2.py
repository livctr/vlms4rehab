from __future__ import annotations
from typing import Optional, Tuple, List, Any, Dict, Union
import numpy as np
from tools.ultralytics_pose import Pose2DStream
from tools.vqa.qwen2_5_vl import Qwen2_5_VL_VQA
import json
from PIL import Image, ImageDraw


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
    
    largest = None
    largest_area = -1.0

    for det in detections_list:
        if not isinstance(det, dict):
            raise TypeError("Each detection must be a dict.")

        raw_bbox = det.get("bbox_2d", det.get("bbox"))
        if raw_bbox is None:
            continue  # skip detections without a bbox

        bbox = _to_float4(raw_bbox)
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if area > largest_area:
            largest_area = area
            largest = (bbox, det.get("label"))

    if largest is None:
        raise KeyError("No valid bounding boxes found.")

    return largest


LOCATE_PROMPT = (
    "Locate the patient as a bounding box in JSON. "
    "If there are multiple people, find all of them."
)

class HandLocator:
    """
    Track the patient and extract wrist/elbow keypoints using a pose model and 
    hand keypoints using a heuristic relying on the ratio of forearm to hand lengths.
    """
    def __init__(
        self,
        stream: Optional[Pose2DStream],
        vqa_model: Optional[Qwen2_5_VL_VQA],
        *,
        hand_wrist_elbow_ratio: float = 0.7  # 0.5 is normal for hand length to forearm length (do 0.7 to see a bit further)
    ):
        """
        Args:
            stream: Pose2DStream instance for 2D pose tracking.
            vqa_model: VQA model instance for person detection prompts.
            hand_wrist_elbow_ratio: Ratio of hand-to-wrist distance to wrist-to-elbow distance.
        """
        self.stream = stream
        self.vqa_model = vqa_model
        self.hand_wrist_elbow_ratio = hand_wrist_elbow_ratio
        self._person_detected = False
    
    def clear(self) -> None:
        self.stream.clear(keep_slot_labels=False, keep_pending_prompts=False)
        self.vqa_model.clear()
        self._person_detected = False

    def _handedness_to_idx(self, handedness: str) -> Tuple[int, int]:
        handedness = handedness.lower()
        assert handedness in ("left", "right")
        K2I = self.stream.KEYPOINT_TO_IDX
        return (K2I["right_elbow"], K2I["right_wrist"]) if handedness == "right" else (K2I["left_elbow"], K2I["left_wrist"])
    
    def process_frames(self, frames: np.ndarray, *, handedness: str, person_locating_prompt: str = None) -> Dict[str, List[np.ndarray]]:
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        # Ensure we are tracking the correct person. May mess up still.
        self.stream.reset_results()
        if not self._person_detected:
            self._person_detected = True
            prompt = person_locating_prompt if person_locating_prompt is not None else LOCATE_PROMPT
            patient_loc_text = self.vqa_model.process_frames(frames[0], context=prompt)
            bbox, label = extract_first_bbox_and_label(patient_loc_text)
            self.stream.add_new_person_to_track(bbox=bbox, label=label)

        # Ensure we get the right elbow/wrist keypoints
        kp_elbow, kp_wrist = self._handedness_to_idx(handedness)

        kps_wrist, kps_elbow, kps_hand = [], [], []
        for i in range(T):
            kps = self.stream.process_frame(frames[i])  # (1, num_person, 17, 3)
            kp = kps[0, 0]
            kps_wrist.append(kp[kp_wrist])
            kps_elbow.append(kp[kp_elbow])

            wx, wy, wc = kp[kp_wrist]
            ex, ey, ec = kp[kp_elbow]

            # check for NaNs
            if (
                np.isnan(wx) or np.isnan(wy) or np.isnan(wc)
                or np.isnan(ex) or np.isnan(ey) or np.isnan(ec)
            ):
                # skip or insert a placeholder
                kps_wrist.append(np.array([np.nan, np.nan, 0.0], dtype=kp.dtype))
                kps_elbow.append(np.array([np.nan, np.nan, 0.0], dtype=kp.dtype))
                kps_hand.append(np.array([np.nan, np.nan, 0.0], dtype=kp.dtype))
                continue

            hx = ex + (wx - ex) * (1.0 + self.hand_wrist_elbow_ratio)  # heuristic for hand position
            hy = ey + (wy - ey) * (1.0 + self.hand_wrist_elbow_ratio)
            hc = min(wc, ec)  # hand confidence = min(wrist, elbow) confidence

            # Clamp everything to image bounds
            wx = max(0, min(W - 1, round(wx)))
            wy = max(0, min(H - 1, round(wy)))
            ex = max(0, min(W - 1, round(ex)))
            ey = max(0, min(H - 1, round(ey)))
            hx = max(0, min(W - 1, round(hx)))
            hy = max(0, min(H - 1, round(hy)))
            kps_hand.append(np.array([hx, hy, hc], dtype=kp.dtype))

        return {
            "wrist": kps_wrist,
            "elbow": kps_elbow,
            "hand": kps_hand,
        }


class HandCropper:
    """
    Different hand-centric cropping methods, applied on top of HandLocator.
    """
    def __init__(
        self,
        *,
        kp_conf_thresh: float = 0.90,
        fast_movement_thresh: int = 12,
        interpolation: str = "middle"  # "middle", "linear", "individual"
    ):
        """
        Args:
            kp_conf_thresh: Minimum confidence for wrist & elbow keypoints to consider valid.
            interpolation: Cropping strategy. "middle" uses the midpoint between first & last frames;
                           "linear" linearly interpolates between first & last frames;
                           "individual" uses each frame's own keypoints if confident for all frames, else full-frames.
                           "mixed" follows "middle" if there is not much motion and "linear" otherwise.
                           If either first/last frames is not confident in "middle" and "linear",
                                full-frame is used.
            fast_movement_thresh: If the L-inf distance between the first and last frame centers
                                    exceeds this threshold times the number of frames minus one,
                                    the status is "FAST MOVEMENT".
        """
        self.kp_conf_thresh = float(kp_conf_thresh)
        self.fast_movement_thresh = int(fast_movement_thresh)
        self._last_center: Optional[Tuple[float, float]] = None
        self._detected = False
        assert interpolation in ("middle", "linear", "individual", "mixed")
        self.interpolation = interpolation

    def clear(self) -> None:
        pass

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

    @staticmethod
    def _bbox_at_center_with_side(
        center: Tuple[float, float], side: int, W: int, H: int
    ) -> Tuple[int, int, int, int]:
        half = side // 2
        cx_i = int(round(max(half, min(W - half, center[0]))))
        cy_i = int(round(max(half, min(H - half, center[1]))))
        x1, y1 = int(cx_i - half), int(cy_i - half)
        x2, y2 = int(cx_i + half), int(cy_i + half)
        # clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        return x1, y1, x2, y2

    def process_frames(self,
                     frames: np.ndarray,
                     *,
                     hand_kps: List[np.ndarray],
                     bbox_side: int = 224
    ):
        """
        Given the list of hand keypoints for a list of frames, produce crops
        on the hand keypoints.

        Returns:
            List of per-frame bounding boxes (x1, y1, x2, y2), ints.
            Flag indicating status: "OK", "ABSTAIN", "FAST MOVEMENT". If "ABSTAIN", 
                all boxes are full-frame. If "FAST MOVEMENT" or "OK", the boxes are cropped
                determined by the interpolation strategy.
        """
        T, H, W, _ = frames.shape
        if self.interpolation == "middle" or self.interpolation == "linear" or self.interpolation == "mixed":
            first = hand_kps[0]
            last = hand_kps[-1]
            ok = (first[2] >= self.kp_conf_thresh) and (last[2] >= self.kp_conf_thresh)
            if not ok:
                full = (0, 0, int(W), int(H))
                return "ABSTAIN", [full for _ in hand_kps]
        
            dist = max(abs(last[0] - first[0]), abs(last[1] - first[1]))
            fast_mvt = dist >= self.fast_movement_thresh * (T - 1)
            use_middle = (self.interpolation == "middle" or (self.interpolation == "mixed" and not fast_mvt))
        
            if use_middle:
                crop_box = self._bbox_at_center_with_side(
                    ((first[0] + last[0]) / 2.0, (first[1] + last[1]) / 2.0),
                    side=bbox_side, W=W, H=H
                )
                return ("FAST MOVEMENT" if fast_mvt else "OK", [crop_box for _ in hand_kps])
            else:
                crop_boxes = [
                    self._bbox_at_center_with_side(
                        (
                            (1 - alpha) * first[0] + alpha * last[0],
                            (1 - alpha) * first[1] + alpha * last[1]
                        ),
                        side=bbox_side, W=W, H=H
                    )
                    for alpha in (i / (T - 1) if T > 1 else 0.0 for i in range(T))
                ]
                return ("FAST MOVEMENT" if fast_mvt else "OK", crop_boxes)

        elif self.interpolation == "individual":
            oks = [
                (kp[2] >= self.kp_conf_thresh) for kp in hand_kps
            ]
            if not all(oks):
                full = (0, 0, int(W), int(H))
                return "ABSTAIN", [full for _ in hand_kps]
        
            crop_boxes = [
                self._bbox_at_center_with_side(
                    (kp[0], kp[1]),
                    side=bbox_side, W=W, H=H
                )
                for kp in hand_kps
            ]
            dist = max(abs(hand_kps[-1][0] - hand_kps[0][0]), abs(hand_kps[-1][1] - hand_kps[0][1]))
            fast_mvt = dist >= self.fast_movement_thresh * (T - 1)
            return ("FAST MOVEMENT" if fast_mvt else "OK", crop_boxes)

        else:
            raise ValueError(f"Unknown interpolation mode: {self.interpolation}")


CONTACT_PROMPT = (
    # "Target objects: {target_objects}\n"
    "Which of the following best describes the hand? Answer directly. \n"
    "- Answer 'Holding.' if it is holding something. \n"
    "- Answer 'Not holding.' if it is not holding something.\n"
    # "- Answer 'Grasping.' if it is fully grasping or holding one of the target objects.\n"
    # "- Answer 'Empty.' if it is not in contact with one of the target objects."
)

# DO_YOU_SEE_TARGET_OBJECTS = (
#     "Target objects: {target_objects}\n"
#     "Do you see any target objects in this frame? Answer the color and identity of the object "
#     "directly if so (e.g. 'red fork', 'blue cup'); otherwise, answer 'None.'\n"
# )

RELEASE_SNAPSHOT_PROMPT = (
    "This is a frame in a video sequence. The hand was holding something in the previous frame. "
    "If the hand is releasing the object in this frame and the object being released is visible, "
    "answer 'Release.'; otherwise, answer 'Other.'\n"
)

GRASP_SNAPSHOT_PROMPT = (
    "This is a frame in a video sequence. The hand was not holding anything in the previous frame. "
    "If the hand is grasping an object in this frame and the object being grasped is visible, "
    "answer 'Grasp.'; otherwise, answer 'Other.'\n"
)


class HandContactDetector:

    def __init__(self, vqa_model: Optional[Qwen2_5_VL_VQA]):
        self.vqa_model = vqa_model
    
    def clear(self) -> None:
        self.vqa_model.clear()
    
    def process_frames(self, frames: np.ndarray, *, target_objects: str) -> List[float]:
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        snapshot_contacts, snapshot_releases, snapshot_grasps = [], [], []
        for i, frame in enumerate(frames):
            prompt = CONTACT_PROMPT.format(target_objects=target_objects)
            c = self.vqa_model.process_frames(frame, context=prompt)
            snapshot_contacts.append(c)

            prompt = RELEASE_SNAPSHOT_PROMPT.format(target_objects=target_objects)
            r = self.vqa_model.process_frames(frame, context=prompt)
            snapshot_releases.append(r)

            prompt = GRASP_SNAPSHOT_PROMPT.format(target_objects=target_objects)
            g = self.vqa_model.process_frames(frame, context=prompt)
            snapshot_grasps.append(g)

            print(c + ' \t\t ' + r + ' \t\t ' + g)
        return snapshot_contacts, snapshot_releases, snapshot_grasps



RELEASE_PROMPT = (
    "This is a chunk in a video sequence. The hand was holding a(n) {target_object} in the previous chunk. "
    "If the hand is visibly releasing the object in this chunk, answer 'Release.'; otherwise, answer 'Other.'\n"
)
GRASP_PROMPT = (
    "This is a chunk in a video sequence. The hand was not holding anything in the previous chunk. "
    "If the hand is grasping an object in this chunk and the object being grasped is visible, "
    "answer 'Grasp.'; otherwise, answer 'Other.'\n"
)
WHICH_OBJECT_PROMPT = (
    "Target objects: {target_objects}\n\n"
    "We just detected a grasp. Answer directly with the color and identity of the object being grasped "
    "(e.g. 'Red fork.', 'Blue cup.').\n"
)


class HandContactReleaseGraspDetector:
    def __init__(self, vqa_model: Optional[Qwen2_5_VL_VQA]):
        self.vqa_model = vqa_model
        self._target_object = ""
    
    def clear(self) -> None:
        self.vqa_model.clear()
    
    def process_frames(self, frames: np.ndarray, *, target_objects: str) -> List[float]:
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        prompt  = RELEASE_PROMPT.format(target_object=self._target_object)
        release = self.vqa_model.process_frames(frames, context=prompt)
        prompt  = GRASP_PROMPT.format(target_objects=target_objects)
        grasp   = self.vqa_model.process_frames(frames, context=prompt)

        if "grasp" in grasp.lower():
            prompt = WHICH_OBJECT_PROMPT.format(target_objects=target_objects)
            which_object = self.vqa_model.process_frames(frames, context=prompt)
        else:
            which_object = "None"

        print(f"Which object: {which_object} out of {target_objects}")
        self._target_object = which_object if "none" not in which_object.lower() else ""

        return [release for _ in range(T)], [grasp for _ in range(T)], [which_object for _ in range(T)]
