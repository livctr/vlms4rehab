"""
State machine implementation for primitives identification. This method breaks
down the problem into three subproblems: idle detection, contact detection, and
transport/stabilize differentiation.

Method:

For a given video, we generate video chunks of 4 frames at 15 FPS (0.25s duration).
We process each chunk sequentially with a state machine that maintains the hand state
and prompts the VLM depending on the current state. Since we observed better performance
when including the previous chunk in the model context, we include it if it is compatible
with the current chunk. The method proceeds as follows for each chunk:

(1) A pose detector extracts elbow and wrist keypoints, which we use to
    approximate the hand location. In addition to the crop,
    we also compute a "pose status" based on the wrist and elbow keypoint
    confidences and absolute displacement in pixel coordinates. The pose status
    is one of "ABSTAIN" (low confidence), "FAST MOVEMENT" (high confidence,
    high displacement), or "OK" (high confidence, low displacement). If the pose
    status is "ABSTAIN", we abstain from making decisions. 
(2) If the pose status is "ABSTAIN", we do not crop around the hand (although this does
    not matter since we will abstain from decisions anyway). If the status is 
    "FAST MOVEMENT", we crop a moving 224x224 window across the chunk that linearly
    interpolates the first and last frames' hand locations. Otherwise, if the status
    is "OK", we crop a still 224x224 window centered at the middle frame's hand
    location. We ensure the crop is within the frame bounds.
(2) Next, we prompt the VLM with the 224x224 cropped chunk to solve the three subproblems:
    (a) Contact detection: We prompt the model grasp/release questions based on the
        current state of the hand.
    (b) Idle detection: We ask a stateless question on whether the hand is idle or not.
    (c) Interaction detection: If the model detects contact, we prompt the model to 
        identify the manner in which the hand is interacting with the object (e.g.
        transport v. stabilize), again based on state.
(3) Based on the model answers, we update the hand state for the next chunk.

We break down the possibilities of the transition table based on the pose status below. Each
cell contains three entries indicating the transition logic for contact, idle, and interaction
detection, respectively. Entry descriptions are below the table. If the cell only contains one
entry, it applies to all three subproblems.

Transition Table:
-----------------
Prev \ Next       | ABSTAIN | FAST_MOVEMENT             | OK
------------------|---------|---------------------------|-----------------------------
ABSTAIN           | Rep     | Rep / Active / Transport  | Recalibrate / IP / TSP
OK                | Rep     | GRP / Active / Transport  | GRP         / IP / TSP
FAST_MOVEMENT     | Rep     | GRP / Active / Transport  | GRP         / IP / TSP

Entry Descriptions:
- **Rep**: We repeat the previous decision w/o prompting the VLM.
- **GRP**: We prompt the VLM a grasp or release question based on the current contact state.
    If the hand is not yet in contact, we ask if it has grasped an object in the current chunk.
        We also ask for the object identity and color based on a list of provided target objects,
        which we use in the prompt for whether the hand releases the object.
    If the hand is in contact, we ask if it has released the object in the current chunk. Because
        the current prompt is biased to miss detections (rather than raise many false alarms), we
        add in an extra check. If the object is visible in the *previous* bounding box location on
        the *current* frame AND the object is not held by a hand, we assume that the object was
        released.
- **Recalibrate**: We prompt the VLM a simple contact question to recalibrate the contact state.
- **Active**: We set the idle state to False (i.e., not idle) w/o prompting the VLM.
- **IP**: We prompt the VLM for whether the hand is idle if the hand is not in contact with an object.
    Otherwise, we set idle to False (i.e., not idle) w/o prompting the VLM.
- **Transport**: We set the interaction state to "transport" w/o prompting the VLM.
- **TSP**: We prompt the VLM for whether the hand is transporting or stabilizing an object.


The organization of this file is as follows:
- Data classes (HandCtx, VideoChunk, VLMProtocol)
- Hand localization (HandLocator, HandCropper)
- CONTACT section (prompts, nodes)
- IDLE section (prompts, nodes)
- INTERACTION section (prompts, nodes)
- Orchestrator (the state machine and main prediction function)
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from lmms_eval.models.model_utils.load_video import load_long_video_decord
from tools.ultralytics_pose import Pose2DStream
from loguru import logger as eval_logger

####################################### DATA CLASSES #######################################

class HandStateStatus:
    OK = "OK"
    ABSTAIN = "ABSTAIN"
    FAST_MOVEMENT = "FAST MOVEMENT"


class InteractionType:
    NONE = ""
    TRANSPORT = "transport"
    STABILIZE = "stabilize"


class HandPrimitives:
    IDLE = "idle"
    REACH = "reach"
    REPOSITION = "reposition"
    TRANSPORT = InteractionType.TRANSPORT
    STABILIZE = InteractionType.STABILIZE
    MOVE = "move"  # placeholder for either reach or reposition



@dataclass
class VideoChunk:
    pose_status: str
    frames: np.ndarray
    bboxes_224x224: List[Tuple[int, int, int, int]]
    start_t: float
    end_t: float


@dataclass
class HandCtx:
    """
    The hand state that moves through the nodes.
    """
    handedness: str  # "left" | "right"
    status: str = HandStateStatus.OK  # "OK" | "ABSTAIN" | "FAST MOVEMENT"

    held_object: str = ""  # Assume the hand can hold only one object at a time.
    idle: bool = True
    interaction: str = InteractionType.NONE  # "transport" | "stabilize" | ""

    # Frames and hand location of the previous chunk.
    frames: np.ndarray = field(default_factory=lambda: np.zeros((0, 224, 224, 3), dtype=np.uint8))
    bboxes_224x224: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def contact(self) -> bool:
        return self.held_object != ""


class VLMProtocol(Protocol):
    def process_frames(self, frames: np.ndarray, prompt: str) -> str:
        """
        Arguments:
            frames: (N, H, W, 3) array of video frames, dtype=uint8, color=RGB
            prompts: list of string prompts, one per frame chunk
        Returns:
            string answer
        """
        ...
    
    def clear(self) -> None:
        """Clear the model state, if any."""
        ...


####################################### Hand Localization #######################################

def _to_float4(
    b: Union[List[Union[int, float]], Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]]
) -> Tuple[float, float, float, float]:
    """Convert bbox coordinates to float tuple."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError(
            f"Expected bbox of length 4, got {type(b).__name__} with "
            f"len={len(b) if hasattr(b, '__len__') else 'N/A'}"
        )
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

def _extract_largest_bbox_and_label(
    detections: Union[str, List[Dict[str, Any]]]
) -> Tuple[Tuple[float, float, float, float], Optional[str]]:
    """Parse the VQA output for bounding boxes. This capability is specific to Qwen2.5(+)-VL."""
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


class HandLocator:
    """
    Track the patient and extract wrist/elbow keypoints using a pose model and 
    hand keypoints using a heuristic relying on the ratio of forearm to hand lengths.
    """
    LOCATE_PROMPT = (
        "Locate the patient as a bounding box in JSON. "
        "If there are multiple people, find all of them."
    )

    def __init__(
        self,
        stream: Optional[Pose2DStream],
        vqa_model: Optional[VLMProtocol],
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
            prompt = person_locating_prompt if person_locating_prompt is not None else self.LOCATE_PROMPT
            patient_loc_text = self.vqa_model.process_frames(frames[0], context=prompt)
            bbox, label = _extract_largest_bbox_and_label(patient_loc_text)
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
        fast_movement_thresh: int = 10,
        min_frames_for_fast_movement: int = 4,
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
        self.min_frames_for_fast_movement = int(min_frames_for_fast_movement)
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
            fast_mvt = (dist >= self.fast_movement_thresh * (T - 1)) and (T >= self.min_frames_for_fast_movement)
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
            fast_mvt = (dist >= self.fast_movement_thresh * (T - 1)) and (T >= self.min_frames_for_fast_movement)
            return ("FAST MOVEMENT" if fast_mvt else "OK", crop_boxes)

        else:
            raise ValueError(f"Unknown interpolation mode: {self.interpolation}")
        


####################################### GRASP/RELEASE SECTION #######################################

def _get_cropped(
    frames: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    cropped_frames = []
    for frame, (x1, y1, x2, y2) in zip(frames, bboxes):
        cropped = frame[y1:y2, x1:x2]  # standard numpy slicing
        cropped_frames.append(cropped)
    return np.stack(cropped_frames, axis=0)


class ProcessingNode(ABC):
    """
    Abstract class for a stateless node object that *operates on* HandCtx
    and returns processed information. The `run` method takes in a fixed set 
    of arguments and can return any type of output.
    """

    def __init__(self, ctx: HandCtx, vlm: VLMProtocol, **fmt):
        self.ctx = ctx
        self.vlm = vlm
        self.fmt = fmt  # formatting kwargs for prompts (e.g., held_object)
    
    def _query_vlm(
        self,
        orig_frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        prompt: str,
        **fmt
    ) -> str:
        """
        Query the VLM with cropped frames whose size is dependent on the
        resolution and a formatted prompt.
        """
        prompt = prompt.format(**fmt)
        cropped_frames = _get_cropped(frames=orig_frames, bboxes=bboxes)
        return self.vlm.process_frames(cropped_frames, prompt)

    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[bool | str | int, Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_state, info) where cur_state is this node's decision/output
            and info is a dictionary of related decision-making information.
        """
        ...


class GraspReleaseProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on* HandContactCtx and returns the next contact
    state.
    """
    BASIC_CONTACT_PROMPT = (
        "Is the hand holding something? Answer 'Yes.' or 'No.' directly.\n"
    )
    GRASP_PROMPT = (
        "This is a chunk from a video sequence. The hand was not holding anything in the previous chunk. "
        "Answer directly: 'Yes.' if the hand grasps an object in this chunk and the object being "
        "grasped is visible; answer 'No.' otherwise.\n"
    )
    WHICH_OBJECT_IN_CONTACT_PROMPT = (
        "Target objects: {target_objects}\n\n"
        "Which object is the hand holding? Answer directly (e.g. 'Fork.', 'Cup.'). ONLY RETURN ONE OBJECT! \n"
    )
    WHAT_COLOR_IS_THE_OBJECT_PROMPT = (
        "What color is the object being held? Answer directly (e.g. 'Red.', 'Blue.').\n"
    )
    RELEASE_PROMPT = (
        "This is a chunk from a video sequence. In the previous chunk, the hand was holding a(n) {target_object}. "
        "Answer directly: 'Yes.' if the hand lets go of it in this chunk and the {target_object} is visible; "
        "answer 'No.' otherwise.\n"
    )
    CHECK_PREV_LOC_1_PROMPT = (
        "Is a(n) {target_object} visible? Answer 'Yes.' or 'No.' directly.\n"
    )
    CHECK_PREV_LOC_2_PROMPT = (
        "Is the {target_object} held by a hand? Answer 'Yes.' or 'No.' directly.\n"
    )

    def __init__(self, ctx: HandCtx, vlm: VLMProtocol, **fmt):
        super().__init__(ctx, vlm, **fmt)
        if "target_objects" not in self.fmt:
            raise ValueError("target_objects must be provided in fmt for ContactStateNode")

    def _repeat(self) -> Tuple[bool, Dict[str, Any]]:
        return (
            self.ctx.contact,
            {"method": "repeat", "result": f"Contact: {self.ctx.contact}->{self.ctx.contact}"}
        )
    
    def _get_held_object(
        self,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> str:
        ans1 = self._query_vlm(frames, bboxes, self.WHICH_OBJECT_IN_CONTACT_PROMPT, **self.fmt).strip().lower()
        held_object = ''.join([ch for ch in ans1 if ch.isalpha() or ch.isspace()])
        if held_object:
            held_object_color = self._query_vlm(frames, bboxes, self.WHAT_COLOR_IS_THE_OBJECT_PROMPT, **self.fmt).strip().lower()
            held_object_color = ''.join([ch for ch in held_object_color if ch.isalpha()])
            held_object = f"{held_object_color} {held_object}".strip()
        return held_object

    def _recalibrate(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        assert pose_status == HandStateStatus.OK
        ans = self._query_vlm(frames, bboxes, self.BASIC_CONTACT_PROMPT, **self.fmt).lower()
        if "yes" in ans:
            held_obj = self._get_held_object(frames, bboxes)
            return (
                held_obj,
                {"method": "recalibrate", "outputs": f"X -> {held_obj} ({ans} to basic contact prompt)"}
            )
        else:
            return (
                "", {"method": "recalibrate", "outputs": f"X -> None ({ans} to basic contact prompt)"}
            )

    def _state_dependent_prompt(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        prev_contact = self.ctx.contact
        if prev_contact:
            fmt = {**self.fmt, "target_object": self.ctx.held_object}
            ans = self._query_vlm(frames, bboxes, self.RELEASE_PROMPT, **fmt).lower()
            if "yes" in ans:
                return (
                    "", {"method": "GRP[R]", "outputs": f"{self.ctx.held_object} -> None ({ans} to release prompt)"}
                )
            else:
                # Possible that the object is just not visible
                # If the object is visible and not held in the previous location, declare released
                prev_bboxes = self.ctx.bboxes_224x224
                ans2 = self._query_vlm(frames, prev_bboxes, self.CHECK_PREV_LOC_1_PROMPT, **fmt).lower()
                if "yes" in ans2:
                    ans3 = self._query_vlm(frames, prev_bboxes, self.CHECK_PREV_LOC_2_PROMPT, **fmt).lower()
                    if "no" in ans3:
                        return (
                            "", {"method": "GRP[R] Override no release",
                                 "outputs": f"{self.ctx.held_object} -> None ({ans} | {ans2} | {ans3})"
                                }
                        )
                else:
                    ans3 = "N/A"
                return (
                    self.ctx.held_object,
                    {"method": "GRP[R]",
                     "outputs": f"{self.ctx.held_object} -> {self.ctx.held_object} ({ans} | {ans2} | {ans3})"
                    }
                )
        else:
            ans = self._query_vlm(frames, bboxes, self.GRASP_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                held_obj = self._get_held_object(frames, bboxes)
                return (
                    held_obj,
                    {"method": "GRP[G]", "outputs": f"None -> {held_obj} ({ans} to grasp prompt)"}
                )
            else:
                return (
                    "", {"method": "GRP[G]", "outputs": f"None -> None ({ans} to grasp prompt)"}
                )

    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[str], Dict[str, Any]]:
        prev_pose_status = self.ctx.status
        should_abstain = (
            pose_status == HandStateStatus.ABSTAIN or
            (prev_pose_status == HandStateStatus.ABSTAIN and pose_status == HandStateStatus.FAST_MOVEMENT)
        )
        if should_abstain:
            return self._repeat()
        should_recalibrate = (
            prev_pose_status == HandStateStatus.ABSTAIN and pose_status == HandStateStatus.OK
        )
        if should_recalibrate:
            return self._recalibrate(pose_status, frames, bboxes)
        return self._state_dependent_prompt(pose_status, frames, bboxes)


####################################### IDLE SECTION #######################################

class IdleProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on *HandCtx* and returns the next idle state
    and info related to the decision.
    """
    PREV_IDLE_PROMPT = (
        "This is a chunk in a video sequence. In the previous chunk, the hand was idle "
        "(i.e. resting on a table or surface without intent to grasp or release an objects). "
        "Answer directly: 'Idle.' if the hand remains idle in this chunk; answer 'Active.' otherwise.\n"
    )
    PREV_NOT_IDLE_PROMPT = (
        "This is a chunk in a video sequence. In the previous chunk, the hand was active "
        "(i.e. it was moving and/or interacting with an object). "
        "Answer directly: 'Active' if the hand remains active in this chunk; answer 'Idle.' otherwise.\n"
    )

    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_idle, info)
        """
        prev_idle = self.ctx.idle
        cur_contact = self.ctx.contact
        if pose_status == HandStateStatus.ABSTAIN:
            return prev_idle, {"method": "repeat", "outputs": f"Idle: {prev_idle} -> {prev_idle}"}
        elif pose_status == HandStateStatus.FAST_MOVEMENT:
            return False, {"method": "fast_movement", "outputs": f"Idle: {prev_idle} -> False"}
        elif pose_status == HandStateStatus.OK:
            if cur_contact:
                return False, {"method": "contact", "outputs": f"Idle: {prev_idle} -> False (contact)"}
            else:
                if prev_idle:
                    ans = self._query_vlm(frames, bboxes, self.PREV_IDLE_PROMPT).lower()
                    if "active" in ans:
                        return False, {"method": "[PREV_IDLE_PROMPT]", "outputs": f"Idle: {prev_idle} -> False. Ans: {ans}"}
                    else:
                        return True, {"method": "[PREV_IDLE_PROMPT]", "outputs": f"Idle: {prev_idle} -> True. Ans: {ans}"}
                else:
                    ans = self._query_vlm(frames, bboxes, self.PREV_NOT_IDLE_PROMPT).lower()
                    if "idle" in ans:
                        return True, {"method": "[PREV_NOT_IDLE_PROMPT]", "outputs": f"Idle: {prev_idle} -> True. Ans: {ans}"}
                    else:
                        return False, {"method": "[PREV_NOT_IDLE_PROMPT]", "outputs": f"Idle: {prev_idle} -> False. Ans: {ans}"}
        else:
            raise ValueError(f"Unknown pose status: {pose_status}")



####################################### INTERACTION SECTION #######################################

class InteractionProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on *HandCtx* and returns the next interaction state
    and info related to the decision.
    """

    TSPrompt = (
        "You are a video analysis expert. Your task is to classify the primary interaction between a hand "
        "and a '{target_object}' in the given video chunk. \n\n"
        
        "## Context Information:\n"
        "- **Target Object:** {target_object}\n"
        "- **Interaction in Previous Chunk:** {prev_interaction_or_none}\n\n"
        
        "## Your Reasoning Process:\n"
        "1. **Observe the Object's Motion:** Is the hand causing the '{target_object}' to move, or is the hand preventing it from moving?\n"
        "2. **Determine the Goal:** Based on the motion, what is the hand's primary goal? Is it to change the object's position/orientation, or to keep it still?\n"
        "3. **Select the Best Category:** Choose one of the three categories below that best describes this primary goal.\n\n"
        
        "## Interaction Categories (Choose one):\n"
        "- **'Transport'**: The hand's primary action is to **actively move or manipulate** the object. The object's position, orientation, or shape is actively changing due to the hand's force. \n"
        "  - **Visual Cues:** Picking up, putting down, sliding, turning, rotating, squeezing, wringing.\n\n"
        
        "- **'Stabilize'**: The hand's primary action is to **prevent the object from moving** or hold it steady. The hand acts as a clamp or support, often against another force. \n"
        "  - **Visual Cues:** Holding an object firmly in place while the other hand performs a task (e.g., holding a jar while twisting the lid); keeping an object from falling or sliding; carefully holding an object still before letting it g - basically minimal movement of hand with the target object in contact\n\n"
        
        "- **'No Interaction'**: The hand is only resting on the object or makes incidental contact without a clear intent to move or stabilize it.\n\n"

        "What best describes the hand's interaction with the {target_object} in this chunk? Answer with a single word only."
    )

    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_idle, info)
        """
        prev_idle = self.ctx.idle
        prev_interaction = self.ctx.interaction
        cur_contact = self.ctx.contact
        if not cur_contact:
            return InteractionType.NONE, {"method": "no_contact", "outputs": "No contact, so no interaction."}
        
        # In contact, could be any of abstain, fast movement, or ok
        if pose_status == HandStateStatus.ABSTAIN:
            return prev_interaction, {"method": "repeat", "outputs": f"Interaction: {prev_interaction} -> {prev_interaction}"}
        elif pose_status == HandStateStatus.FAST_MOVEMENT:
            return InteractionType.TRANSPORT, {"method": "fast_movement", "outputs": f"Interaction: {prev_interaction} -> transport"}
        elif pose_status == HandStateStatus.OK:
            prev_interaction_or_none = prev_interaction if prev_interaction != InteractionType.NONE else "none (the hand just made contact)"
            ans = self._query_vlm(
                frames,
                bboxes,
                self.TSPrompt,
                target_object=self.ctx.held_object,
                prev_interaction_or_none=prev_interaction_or_none
            ).lower()
            if "stabilize" in ans:
                return (
                    InteractionType.STABILIZE,
                    {"method": "TS", "outputs": f"Interaction: {prev_interaction} -> stabilize. Ans: {ans}"}
                )
            else:  # By default, we classify as transport if either 'Transport' or 'No interaction' is detected
                return (
                    InteractionType.TRANSPORT,
                    {"method": "TS", "outputs": f"Interaction: {prev_interaction} -> transport. Ans: {ans}"}
                )
        else:
            raise ValueError(f"Unknown pose status: {pose_status}")


####################################### ORCHESTRATION SECTION #######################################


def _bboxes_are_compatible(
    bboxes1: List[Tuple[int, int, int, int]],
    bboxes2: List[Tuple[int, int, int, int]]
):
    # Only need to check the first bbox of each list
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return False
    return bboxes1[0][2] - bboxes1[0][0] == bboxes2[0][2] - bboxes2[0][0] and \
           bboxes1[0][3] - bboxes1[0][1] == bboxes2[0][3] - bboxes2[0][1]


class HandStateMachine:
    def __init__(self, *, handedness: str, vlm: VLMProtocol, target_objects: str = ""):
        self.ctx = HandCtx(handedness=handedness)
        self.target_objects = target_objects
        self.grpn = GraspReleaseProcessingNode(self.ctx, vlm, target_objects=target_objects)
        self.ipn  = IdleProcessingNode(self.ctx, vlm, target_objects=target_objects)
        self.tspn = InteractionProcessingNode(self.ctx, vlm, target_objects=target_objects)

    def step(
        self, chunk: VideoChunk
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        
        # If the bboxes are the same size, we can concatenate the frames
        # We expect the chunk generator to generate bboxes of the same shape within each chunk
        # We only have to check the first bboxes of each chunk
        if _bboxes_are_compatible(self.ctx.bboxes_224x224, chunk.bboxes_224x224):
            prev_and_cur_frames_224 = np.concatenate([self.ctx.frames, chunk.frames], axis=0)
            prev_and_cur_bboxes_224 = self.ctx.bboxes_224x224 + chunk.bboxes_224x224
        else:
            prev_and_cur_frames_224 = chunk.frames
            prev_and_cur_bboxes_224 = chunk.bboxes_224x224
        
        # CONTACT
        held_obj, held_obj_info = self.grpn.run(
            chunk.pose_status, prev_and_cur_frames_224, prev_and_cur_bboxes_224
        )
        self.ctx.held_object = held_obj   # Important to update before IDLE and INTERACTION

        # IDLE
        idle, idle_info = self.ipn.run(chunk.pose_status, chunk.frames, chunk.bboxes_224x224)
        self.ctx.idle = idle

        # INTERACTION (transport/stabilize/none)
        interaction, interaction_info = self.tspn.run(chunk.pose_status, prev_and_cur_frames_224, chunk.bboxes_224x224)
        self.ctx.interaction = interaction

        self.ctx.status = chunk.pose_status
        self.ctx.frames = chunk.frames
        self.ctx.bboxes_224x224 = chunk.bboxes_224x224

        if self.ctx.contact:
            pred = interaction  # either transport or stabilize
        else:
            if idle:
                pred = HandPrimitives.IDLE
            else:
                pred = HandPrimitives.MOVE  # or reposition, but we do not distinguish yet
        
        info = {
            "held_object": held_obj,
            "idle_info": str(idle_info),
            "held_obj_info": str(held_obj_info),
            "interaction_info": str(interaction_info)
        }
        return pred, info

    @property
    def context(self) -> HandCtx:
        return self.ctx


def _get_target_objects(video_path: str) -> str:
    video_path = video_path.lower()
    if "face" in video_path:
        return "Target objects: washcloth, handle, tub"
    elif "deodrant" in video_path or "deodorant" in video_path:  # include both
        return "Target objects: deodorant tube, deodorant cap"
    elif "combing" in video_path:
        return "Target objects: comb"
    elif "glasses" in video_path:
        return "Target objects: glasses"
    elif "feeding" in video_path:
        return "Target objects: paper plate, fork, knife, re-sealable plastic bag, bread, margarine"
    elif "drinking" in video_path:
        return "Target objects: water bottle, bottle cap, cup"
    elif "brushing" in video_path:
        return "Target objects: toothpaste, toothbrush, handle"
    else:
        return "Target objects: toilet paper roll"


def predict_with_state_machine(
    video_path: str,
    handedness: str,
    vlm: VLMProtocol,
    pose_stream: Pose2DStream,
    max_frames_num: int = 4,
    sampling_strategy: str = "dense",
    overlap_frames_num: int = 0,
    sampling_fps: int = 15,
    num_frames_for_idle: int = 4,
    min_frames_for_reach_reposition: int = 8,
) -> Tuple[List[str], List[float], Dict[str, Any]]:
    """
    Arguments:
        video_path: path to the video file
        handedness: "left" | "right", which hand to track
        vlm: the vision-language model that implements VLMProtocol
        pose_stream: the 2D pose predictor
        max_frames_num: number of frames per chunk
        sampling_strategy: "dense" | "uniform"
        overlap_frames_num: number of overlapping frames between chunks (only for "dense"). Keep at 0.
        sampling_fps: fps to sample the video at
        num_frames_for_idle: number of frames where idle is detected to be considered idle
        num_frames_reach_reposition: minimum number of frames to split reposition-reach

    Returns a list of primitives (one per frame), their timestamps, and detailed info.

    Because this state machine is yet to discern between stabilize and transport,
    we return 'idle', 'reach', 'reposition', and 'transport'.
    """

    target_objects = _get_target_objects(video_path)
    machine = HandStateMachine(handedness=handedness, vlm=vlm, target_objects=target_objects)
    hand_locator = HandLocator(pose_stream, vlm)
    hand_cropper = HandCropper(
        kp_conf_thresh=0.90, fast_movement_thresh=8, interpolation="mixed"
    )
    hand_locator.clear()
    hand_cropper.clear()

    times, infos = [], {}

    for frames, start_t, end_t in load_long_video_decord(
        video_path,
        max_frames_num=max_frames_num,
        sampling_strategy=sampling_strategy,
        overlap_frames_num=overlap_frames_num,
        sampling_fps=sampling_fps,
        force_sample=False,
        ret_idx=False,
    ):

        kp = hand_locator.process_frames(frames, handedness=handedness)
        kp_wrist, kp_elbow, kp_hand = kp["wrist"], kp["elbow"], kp["hand"]
        pose_status, boxes_224x224 = hand_cropper.process_frames(frames, hand_kps=kp_hand, bbox_side=224)

        chunk = VideoChunk(
            pose_status=pose_status,
            frames=frames,
            bboxes_224x224=boxes_224x224,
            start_t=start_t,
            end_t=end_t,
        )
        prim, info = machine.step(chunk)

        num_frames = len(frames)
        times.extend(np.linspace(start_t, end_t, num_frames, endpoint=False).tolist())
        info["status"] = [pose_status] * num_frames
        info["prim"] = [prim] * num_frames
        info["kps_wrist"] = [kp_wrist[i] for i in range(num_frames)]
        info["kps_elbow"] = [kp_elbow[i] for i in range(num_frames)]
        info["kps_hand"] = [kp_hand[i] for i in range(num_frames)]
        info["bboxes_224x224"] = [boxes_224x224[i] for i in range(num_frames)]
        for key in info:
            if key not in infos:
                infos[key] = []
            if type(info[key]) is list:
                infos[key].extend(info[key])
            else:
                infos[key].extend([info[key]] * num_frames)

        eval_logger.info(
            f"{start_t:.2f}-{end_t:.2f}s | {pose_status} | {prim} | {infos['held_object']} | {infos['idle_info']} | {infos['held_obj_info']} | {infos['interaction_info']}"
        )

    unprocessed_prims = infos["prim"]
    N = len(unprocessed_prims)
    idles = [unprocessed_prims[i] == HandPrimitives.IDLE for i in range(N)]
    contacts = [(unprocessed_prims[i] == HandPrimitives.TRANSPORT) or (unprocessed_prims[i] == HandPrimitives.STABILIZE) for i in range(N)]

    # -------- Postprocessing -------

    # 1) Set idle, reach, and reposition to UNK.
    prims = [prim if contact else "UNK" for prim, contact in zip(unprocessed_prims, contacts)]

    # 2) Verify the idle: we need at least `num_frames_for_idle` consecutive frames to declare an idle.
    #    Set idle.
    num_prev_idle = [0] * N
    num_prev_idle[0] = 1 if idles[0] else 0
    for i in range(1, N):
        if idles[i]:
            num_prev_idle[i] = num_prev_idle[i-1] + 1
        else:
            num_prev_idle[i] = 0
    verified_idles = [0] * N
    j = N - 1
    while j >= 0:
        if num_prev_idle[j] >= num_frames_for_idle:
            for k in range(j, j - num_prev_idle[j], -1):
                verified_idles[k] = 1
            j -= num_prev_idle[j]
        else:
            j -= 1
    prims = [
        HandPrimitives.IDLE if (not contact and verified_idle) else prim
        for prim, verified_idle, contact in zip(prims, verified_idles, contacts)
    ]

    # 3) Determine between reach, reposition, and reposition-reach
    contact_in_future = [False] * N
    contact_state = False
    for i in range(N - 1, -1, -1):
        if contact_state:
            if prims[i] == HandPrimitives.IDLE:
                contact_state = False
        else:
            if contacts[i]:
                contact_state = True
        contact_in_future[i] = contact_state
    contact_in_past = [False] * N
    contact_state = False
    for i in range(N):
        if contact_state:
            if prims[i] == HandPrimitives.IDLE:
                contact_state = False
        else:
            if contacts[i]:
                contact_state = True
        contact_in_past[i] = contact_state

    # Go through prims
    # - If no future contact -> reposition
    # - If no prior contact and future contact -> reach
    # - If prior contact and future contact -> reposition-reach
    for i in range(N):
        if prims[i] == "UNK":
            if not contact_in_future[i]:
                prims[i] = HandPrimitives.REPOSITION
            elif not contact_in_past[i]:
                prims[i] = HandPrimitives.REACH
    # The list is now "idle", "reach", "reposition", "transport", and "UNK"
    # Still UNK? Must be reposition-reach
    # Fill in UNK spans with reposition-reach
    i = 0
    while i < N:
        if prims[i] != "UNK":
            i += 1
            continue

        j = i
        while j < N and prims[j] == "UNK":
            j += 1
        
        L = j - i  # length of UNK span: 4 = 2, 2. 3 = 2, 1
        if L < min_frames_for_reach_reposition:
            prims[i:j] = [HandPrimitives.REACH] * L  # too short, just reach
        else:
            first_half = (L - 1) // 2 + 1
            second_half = L - first_half
            prims[i : i + first_half] = [HandPrimitives.REPOSITION] * first_half
            prims[i + first_half : j] = [HandPrimitives.REACH] * second_half

        i = j

    # 4) Smooth out quick idles -> contacts and contacts -> idles 
    # with a reach/reposition
    side = max_frames_num // 2
    for i in range(1, N):
        if prims[i-1] == HandPrimitives.IDLE and prims[i] == HandPrimitives.TRANSPORT:
            for j in range(max(0, i-side), min(N, i+side)):
                prims[j] = HandPrimitives.REACH

        if prims[i-1] == HandPrimitives.TRANSPORT and prims[i] == HandPrimitives.IDLE:
            for j in range(max(0, i-side), min(N, i+side)):
                prims[j] = HandPrimitives.REPOSITION

    return prims, times, infos
