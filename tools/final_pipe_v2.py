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
    is one of "ABSTAIN" (low confidence), "FAST" (high confidence,
    high displacement), or "OK" (high confidence, low displacement). If the pose
    status is "ABSTAIN", we abstain from making decisions. 
(2) If the pose status is "ABSTAIN", we do not crop around the hand (although this does
    not matter since we will abstain from decisions anyway). If the status is 
    "FAST", we crop a moving 224x224 window across the chunk that linearly
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
    fast_mvt: bool
    hand_reference: str
    frames: np.ndarray
    bboxes: List[Tuple[int, int, int, int]]
    start_t: float
    end_t: float


@dataclass
class HandCtx:
    """
    The hand state that moves through the nodes.
    """
    handedness: str  # "left" | "right"

    # Whether the hand is moving fast, as identified by the pose model
    fast_mvt: bool = False
    # How the hand is referred to in the prompt, as identified by the cropper
    hand_reference: str = "the hand in focus"

    held_object: str = ""  # Assume the hand can hold only one object at a time.
    idle: bool = True
    interaction: str = InteractionType.NONE  # "transport" | "stabilize" | ""

    # Frames and hand location of the previous chunk.
    frames: np.ndarray = field(default_factory=lambda: np.zeros((0, 224, 224, 3), dtype=np.uint8))
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

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
    
    def _clamp_xy(self, x: float, y: float, W: int, H: int) -> Tuple[int, int]:
        """Round then clamp (x, y) to [0..W-1], [0..H-1]."""
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0: xi = 0
        elif xi >= W: xi = W - 1
        if yi < 0: yi = 0
        elif yi >= H: yi = H - 1
        return xi, yi

    def _nan_in_triplet(self, triplet: np.ndarray) -> bool:
        """True if any NaN in a (x, y, conf) triplet."""
        return np.isnan(triplet).any()

    def _append_placeholder(
        self,
        out: Dict[str, Dict[str, List[np.ndarray]]],
        side: str,
        dtype: np.dtype,
    ) -> None:
        """Append placeholder (nan, nan, 0.0) for wrist/elbow/hand."""
        ph = np.array([np.nan, np.nan, 0.0], dtype=dtype)
        out[side]["wrist"].append(ph)
        out[side]["elbow"].append(ph)
        out[side]["hand"].append(ph)

    def _append_keypoints(
        self,
        out: Dict[str, Dict[str, List[np.ndarray]]],
        side: str,
        *,
        wrist: Tuple[int, int, float],
        elbow: Tuple[int, int, float],
        hand: Tuple[int, int, float],
        dtype: np.dtype,
    ) -> None:
        """Append concrete wrist/elbow/hand triplets."""
        out[side]["wrist"].append(np.asarray(wrist, dtype=dtype))
        out[side]["elbow"].append(np.asarray(elbow, dtype=dtype))
        out[side]["hand"].append(np.asarray(hand, dtype=dtype))

    def _compute_hand_from_elbow_wrist(
        self,
        elbow: np.ndarray,
        wrist: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Heuristic hand = elbow + (wrist - elbow) * (1 + ratio).
        Conf = min(conf_wrist, conf_elbow).
        """
        ex, ey, ec = map(float, elbow)
        wx, wy, wc = map(float, wrist)
        ratio = 1.0 + float(self.hand_wrist_elbow_ratio)
        hx = ex + (wx - ex) * ratio
        hy = ey + (wy - ey) * ratio
        hc = float(min(wc, ec))
        return hx, hy, hc

    def _process_side(
        self,
        out: Dict[str, Dict[str, List[np.ndarray]]],
        side: str,
        kp: np.ndarray,               # shape (17, 3)
        idx_elbow: int,
        idx_wrist: int,
        W: int,
        H: int,
    ) -> None:
        """Read elbow/wrist, handle NaNs, compute hand, clamp, append."""
        dtype = kp.dtype
        elbow = kp[idx_elbow]  # (3,)
        wrist = kp[idx_wrist]  # (3,)

        if self._nan_in_triplet(elbow) or self._nan_in_triplet(wrist):
            self._append_placeholder(out, side, dtype)
            return

        hx, hy, hc = self._compute_hand_from_elbow_wrist(elbow, wrist)
        ex_i, ey_i = self._clamp_xy(elbow[0], elbow[1], W, H)
        wx_i, wy_i = self._clamp_xy(wrist[0], wrist[1], W, H)
        hx_i, hy_i = self._clamp_xy(hx, hy, W, H)

        self._append_keypoints(
            out,
            side,
            wrist=(wx_i, wy_i, float(wrist[2])),
            elbow=(ex_i, ey_i, float(elbow[2])),
            hand=(hx_i, hy_i, hc),
            dtype=dtype,
        )

    def process_frames(
        self,
        frames: np.ndarray,
        *,
        person_locating_prompt: str = None,
    ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        """
        Process frames and return left/right wrist, elbow, and (heuristic) hand keypoints.

        Returns
        -------
        {
        "left":  { "wrist": [np.ndarray(...), ...],
                    "elbow": [...],
                    "hand":  [...] },
        "right": { "wrist": [...],
                    "elbow": [...],
                    "hand":  [...] }
        }
        """
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        # Ensure we are tracking the correct person (same as before).
        self.stream.reset_results()
        if not getattr(self, "_person_detected", False):
            self._person_detected = True
            prompt = person_locating_prompt if person_locating_prompt is not None else self.LOCATE_PROMPT
            patient_loc_text = self.vqa_model.process_frames(frames[0], context=prompt)
            bbox, label = _extract_largest_bbox_and_label(patient_loc_text)
            self.stream.add_new_person_to_track(bbox=bbox, label=label)

        # Indices for both sides
        l_elbow_idx, l_wrist_idx = self._handedness_to_idx("left")
        r_elbow_idx, r_wrist_idx = self._handedness_to_idx("right")

        # Output buffers
        out = {
            "left":  {"wrist": [], "elbow": [], "hand": []},
            "right": {"wrist": [], "elbow": [], "hand": []},
        }

        for i in range(T):
            kps = self.stream.process_frame(frames[i])  # (1, num_person, 17, 3)
            kp = kps[0, 0]                              # (17, 3)
            self._process_side(out, "left",  kp, l_elbow_idx, l_wrist_idx, W, H)
            self._process_side(out, "right", kp, r_elbow_idx, r_wrist_idx, W, H)
        return out


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


def _get_first_last_conf(
    kps: Dict[str, Dict[str, List[np.ndarray]]],
    handedness: str,
    conf_thresh: float
) -> Tuple[np.ndarray, np.ndarray, bool, float, float]:
    """Return first/last keypoints, confidence booleans, and usable flag."""
    first = kps[handedness]["hand"][0]
    last = kps[handedness]["hand"][-1]
    conf_first, conf_last = float(first[2]), float(last[2])
    ok = (conf_first >= conf_thresh) and (conf_last >= conf_thresh)
    return first, last, ok, conf_first, conf_last


def _midpoint(pt0: np.ndarray, pt1: np.ndarray) -> Tuple[float, float]:
    """Return midpoint between two (x, y) coordinates."""
    return ((pt0[0] + pt1[0]) / 2.0, (pt0[1] + pt1[1]) / 2.0)


def _hand_relative_position(
    cur_mid: Tuple[float, float],
    other_mid: Tuple[float, float],
    cur_conf: float,
    other_conf: float,
    close_thresh: int = 56
) -> str:
    """Determine relative text: close→front/back else LEFT/RIGHT/ABOVE/BELOW."""
    dx = float(cur_mid[0] - other_mid[0])
    dy = float(cur_mid[1] - other_mid[1])

    if max(abs(dx), abs(dy)) <= close_thresh:
        return "the hand in front" if cur_conf >= other_conf else "the hand in the back"

    if dx <= -abs(dy):
        where = "on the LEFT SIDE"
    elif dx >= abs(dy):
        where = "on the RIGHT SIDE"
    elif dy <= -abs(dx):
        where = "ABOVE"
    else:
        where = "BELOW"

    return f"the hand {where} relative to the camera"


class HandCropper:
    """
    Different hand-centric cropping methods, applied on top of HandLocator.
    """
    def __init__(
        self,
        *,
        kp_conf_thresh: float,
        fast_movement_thresh: int,
        min_frames_for_fast_movement: int = 4,
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
                                    the status is "FAST".

        Note: Cropping strategy:
        X: If the first and last frames are confident (both elbow/wrist surpass kp_conf_thresh)
        Y: If the distance between the first and last frames exceeds fast_movement_thresh * (T - 1)
        - If X and Y, we use a moving crop that linearly interpolates between the first and last frames.
        - If X and not Y, we use a still crop centered at the midpoint between the first and last frames.
        - If not X, we don't crop.
        """
        self.kp_conf_thresh = float(kp_conf_thresh)
        self.fast_movement_thresh = int(fast_movement_thresh)
        self.min_frames_for_fast_movement = int(min_frames_for_fast_movement)
        self._last_center: Optional[Tuple[float, float]] = None
        self._detected = False

    def clear(self) -> None:
        pass

    def process_frames(
        self,
        frames: np.ndarray,
        handedness: str,
        kps: Dict[str, Dict[str, List[np.ndarray]]],
        *,
        bbox_side: int = 224,
    ) -> Tuple[bool, str, List[Tuple[int, int, int, int]]]:
        """
        Compute per-frame crops around the target hand based on keypoint confidence and motion,
        then produce a textual hand_reference describing which hand is being referred to.

        Returns:
            moving_crop: bool indicating whether the crop is moving across frames.
            ref_text:    A short phrase per the decision rules.
            crop_boxes:  List[(x1, y1, x2, y2)] for each frame.
        """
        T, H, W, _ = frames.shape

        cur_first, cur_last, cur_ok, cur_conf_first, cur_conf_last = _get_first_last_conf(
            kps, handedness, self.kp_conf_thresh
        )
        other = "left" if handedness == "right" else "right"
        other_first, other_last, other_ok, other_conf_first, other_conf_last = _get_first_last_conf(
            kps, other, self.kp_conf_thresh
        )

        # --- Step 1: Crop computation (same logic as before) ---
        if not cur_ok:
            full = (0, 0, int(W), int(H))
            crop_boxes = [full for _ in range(T)]
            moving_crop = False
        else:
            dist = max(abs(cur_last[0] - cur_first[0]), abs(cur_last[1] - cur_first[1]))
            fast_mvt = (dist >= self.fast_movement_thresh * max(1, (T - 1))) and (
                T >= self.min_frames_for_fast_movement
            )
            moving_crop = bool(fast_mvt)

            if fast_mvt:
                crop_boxes = [
                    _bbox_at_center_with_side(
                        (
                            (1 - alpha) * cur_first[0] + alpha * cur_last[0],
                            (1 - alpha) * cur_first[1] + alpha * cur_last[1],
                        ),
                        side=bbox_side,
                        W=W,
                        H=H,
                    )
                    for alpha in (i / (T - 1) if T > 1 else 0.0 for i in range(T))
                ]
            else:
                mid = _midpoint(cur_first, cur_last)
                box = _bbox_at_center_with_side(mid, side=bbox_side, W=W, H=H)
                crop_boxes = [box for _ in range(T)]

        # --- Step 2: Textual hand_reference ---

        # Hand is not usable
        if not cur_ok:
            if other_ok:
                ref_text = "the occluded hand of the patient"
            else:
                ref_text = f"the patient's {handedness} hand"
            return moving_crop, ref_text, crop_boxes
        
        other_mid = _midpoint(other_first, other_last)
        other_inframe = _bbox_contains_points(crop_boxes[0], (other_mid,))  # Frames are the same if not moving_crop
        if moving_crop or not other_ok or not other_inframe:
            # We assume in these two cases that it is sufficiently obvious for the 
            # VLM to know which hand we mean.
            # (1) In a moving crop, we are mainly focusing on the moving hand.
            # (2) If the other hand is not usable and the current one is, it is likely
            #     that the current hand is the one that the VLM will pick up on.
            # (3) The other hand is simply not in frame.
            ref_text = "the hand being tracked" if moving_crop else "the hand in focus"
            return moving_crop, ref_text, crop_boxes

        # Both hands are in frame and confident    
        cur_mid = _midpoint(cur_first, cur_last)
        cur_mean_conf = 0.5 * (cur_conf_first + cur_conf_last)
        other_mean_conf = 0.5 * (other_conf_first + other_conf_last)
        ref_text = _hand_relative_position(cur_mid, other_mid, cur_mean_conf, other_mean_conf)
        return moving_crop, ref_text, crop_boxes


####################################### IDLE SECTION #######################################

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
        fast_mvt: bool,
        hand_reference: str,
        orig_frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        prompt: str,
        **fmt,
    ) -> str:
        """
        Query the VLM with cropped frames whose size is dependent on the
        resolution and a formatted prompt.
        """
        fmt = {**fmt, "the_referred_hand": hand_reference}
        prompt = prompt.format(**fmt)
        cropped_frames = _get_cropped(frames=orig_frames, bboxes=bboxes)
        return self.vlm.process_frames(cropped_frames, prompt)

    def run(
        self,
        fast_mvt: bool,
        hand_reference: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[bool | str | int, Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_state, info) where cur_state is this node's decision/output
            and info is a dictionary of related decision-making information.
        """
        ...


class IdleProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on *HandCtx* and returns the next idle state
    and info related to the decision.
    """
    IDLE_PROMPT = (
        "Is {the_referred_hand} idle in this clip? Answer only 'IDLE' or 'ACTIVE'. \n"
        "(Idle) {the_referred_hand} is still or barely moving, and its fingers and wrist appear relaxed. "
        "Answer 'IDLE' if the hand looks to be at rest without holding an object, "
        "even if it is near an object or in the air. \n"
        "(Active) {the_referred_hand} is moving with purpose — its fingers or wrist are tensed, changing position, "
        "or interacting with an object through reaching, grasping, pressing, turning, adjusting, squeezing, or holding. "
        "If the hand looks like it is holding an object, answer 'ACTIVE'."
    )

    def run(
        self,
        fast_mvt: bool,
        hand_reference: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_idle, info)
        """
        if fast_mvt:
            return False, {"method": "IP[FAST]", "outputs": "False (fast movement)"}
        ans = self._query_vlm(fast_mvt, hand_reference, frames, bboxes, self.IDLE_PROMPT, **self.fmt).lower()
        cur_idle = "idle" in ans
        return cur_idle, {"method": "IP", "outputs": f"{cur_idle} ({ans})"}



####################################### GRASP/RELEASE SECTION #######################################

class GraspReleaseProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on* HandContactCtx and returns the next contact
    state.
    """
    GRASP_PROMPT = (
        "Target objects: {target_objects}\n\n"
        "This clip follows one where {the_referred_hand} hasn't made contact with a target object yet. "
        "Question: Does {the_referred_hand} actively make contact with a target object in this clip? "
        "If so, either answer with the name of the object (e.g. 'Fork.', 'Cup.'). "
        "Otherwise, answer 'Not yet.' (e.g. if the hand is just resting on the object rather than supporting it)."
    )
    RELEASE_PROMPT = (
        "This clip follows one where {the_referred_hand} was holding a(n) {target_object}. "
        "Answer directly: 'Yes.' if the hand lets go of it in this chunk and the {target_object} is visible; "
        "answer 'No.' otherwise.\n"
    )
    RELEASE_CHECK_1_PROMPT = (
        "Is a(n) {target_object} visible? Answer 'Yes.' or 'No.' directly.\n"
    )
    RELEASE_CHECK_2_PROMPT = (
        "Is the {target_object} being held by a hand? Answer 'Yes.' or 'No.' directly.\n"
    )

    def __init__(self, ctx: HandCtx, vlm: VLMProtocol, **fmt):
        super().__init__(ctx, vlm, **fmt)
        if "target_objects" not in self.fmt:
            raise ValueError("target_objects must be provided in fmt for ContactStateNode")

    def _release_check(
        self,
        fast_mvt: bool,
        hand_reference: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[bool, str]:
        """Release check after 'release.' """
        if len(self.ctx.bboxes) == 0:
            return True, "Cannot check"
        # Only allow release if CHECK_1 is yes and CHECK_2 is no.
        fmt = {**self.fmt, "target_object": self.ctx.held_object}
        prev_bboxes = [self.ctx.bboxes[0] for _ in range(len(frames))]
        ans1 = self._query_vlm(
            fast_mvt, hand_reference, frames, prev_bboxes, self.RELEASE_CHECK_1_PROMPT, **fmt
        ).strip().lower()
        if "yes" in ans1:
            ans2 = self._query_vlm(
                fast_mvt, hand_reference, frames, prev_bboxes, self.RELEASE_CHECK_2_PROMPT, **fmt
            ).strip().lower()
        else:
            ans2 = "N/A"
        release_confirmed = ("yes" in ans1) and ("no" in ans2)
        return release_confirmed, f"Checked released ({ans1} | {ans2})"

    def run(
        self,
        fast_mvt: bool,
        hand_reference: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        cur_idle = self.ctx.idle
        if cur_idle:
            return "", {"method": "GRP[IDLE]", "outputs": "N/A"}
        prev_contact = self.ctx.contact

        if prev_contact:
            held_obj = self.ctx.held_object

            fmt = {**self.fmt, "target_object": held_obj}
            released = self._query_vlm(
                fast_mvt, hand_reference, frames, bboxes, self.RELEASE_PROMPT, **fmt
            ).lower()
            if "yes" in released:
                return (
                    "", {"method": "GRP[R]", "outputs": f"Released ({released})"}
                )
            
            released_checked, released_checked_info = self._release_check(
                fast_mvt, hand_reference, frames, bboxes
            )
            if released_checked:
                return (
                    "", {"method": "GRP[R]", "outputs": f"Released ({released} | {released_checked_info})"}
                )
            return (
                held_obj, {"method": "GRP[R]", "outputs": f"Held ({released} | {released_checked_info})"}
            )

        else:
            ans = self._query_vlm(fast_mvt, hand_reference, frames, bboxes, self.GRASP_PROMPT, **self.fmt).lower()
            if "not yet" in ans:
                held_obj = ""
            else:
                held_obj = "".join([ch for ch in ans if ch.isalpha() or ch.isspace()]).strip()
            return (
                held_obj,
                {"method": "GRP[G]", "outputs": f"{held_obj} ({ans})"}
            )


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
        fast_mvt: bool,
        hand_reference: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_idle, info)
        """
        if fast_mvt:
            return InteractionType.TRANSPORT, {"method": "TS[FAST]", "outputs": "transport"}
        return InteractionType.TRANSPORT, {"method": "TS[NC]", "outputs": "N/A"}


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
        """Processing order matters."""

        idle, idle_info = self.ipn.run(
            chunk.fast_mvt, chunk.hand_reference, chunk.frames, chunk.bboxes
        )
        self.ctx.idle = idle

        held_obj, held_obj_info = self.grpn.run(
            chunk.fast_mvt, chunk.hand_reference, chunk.frames, chunk.bboxes
        )
        self.ctx.held_object = held_obj

        # Interaction
        interaction, interaction_info = self.tspn.run(
            chunk.fast_mvt, chunk.hand_reference, chunk.frames, chunk.bboxes
        )
        self.ctx.interaction = interaction

        self.ctx.fast_mvt = chunk.fast_mvt
        self.ctx.hand_reference = chunk.hand_reference
        self.ctx.frames = chunk.frames
        self.ctx.bboxes = chunk.bboxes

        if idle:
            pred = HandPrimitives.IDLE
        else:
            if self.ctx.contact:
                pred = interaction
            else:
                pred = HandPrimitives.MOVE  # or reposition, but we do not distinguish yet

        info = {
            "contact": self.ctx.contact,
            "idle": self.ctx.idle,
            "interaction": self.ctx.interaction,
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
        return "Target objects: washcloth, faucet handle, tub"
    elif "deodrant" in video_path or "deodorant" in video_path:  # include both
        return "Target objects: deodorant tube, deodorant cap"
    elif "combing" in video_path:
        return "Target objects: comb"
    elif "glasses" in video_path:
        return "Target objects: glasses"
    elif "feeding" in video_path:
        return "Target objects: paper plate, fork, knife, re-sealable plastic bag, bread, margarine"
    elif "drinking" in video_path:
        return "Target objects: water bottle, water bottle cap, cup"
    elif "brushing" in video_path:
        return "Target objects: toothpaste, toothbrush, faucet handle"
    else:
        return "Target objects: toilet paper roll"


def predict_with_state_machine(
    video_path: str,
    handedness: str,
    vlm: VLMProtocol,
    pose_stream: Pose2DStream,
    chunk_max_frames: int = 4,
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
        chunk_max_frames: number of frames per chunk
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
    hand_cropper = HandCropper(kp_conf_thresh=0.90, fast_movement_thresh=8)
    hand_locator.clear()
    hand_cropper.clear()

    times, infos = [], {}

    for frames, start_t, end_t in load_long_video_decord(
        video_path,
        max_frames_num=chunk_max_frames,
        sampling_strategy=sampling_strategy,
        overlap_frames_num=overlap_frames_num,
        sampling_fps=sampling_fps,
        force_sample=False,
        ret_idx=False,
    ):

        kp = hand_locator.process_frames(frames)
        fast_mvt, hand_reference, boxes = hand_cropper.process_frames(frames, handedness, kp, bbox_side=224)
        chunk = VideoChunk(
            fast_mvt=fast_mvt,
            hand_reference=hand_reference,
            frames=frames,
            bboxes=boxes,
            start_t=start_t,
            end_t=end_t,
        )
        prim, info = machine.step(chunk)

        num_frames = len(frames)
        times.extend(np.linspace(start_t, end_t, num_frames, endpoint=False).tolist())
        info["status"] = [hand_reference] * num_frames
        info["raw_prims"] = [prim] * num_frames
        info["kps_wrist"] = [kp[handedness]["wrist"][i] for i in range(num_frames)]
        info["kps_elbow"] = [kp[handedness]["elbow"][i] for i in range(num_frames)]
        info["kps_hand"] = [kp[handedness]["hand"][i] for i in range(num_frames)]
        other = "left" if handedness == "right" else "right"
        info["kps_wrist_other"] = [kp[other]["wrist"][i] for i in range(num_frames)]
        info["kps_elbow_other"] = [kp[other]["elbow"][i] for i in range(num_frames)]
        info["kps_hand_other"] = [kp[other]["hand"][i] for i in range(num_frames)]
        info["bboxes"] = [boxes[i] for i in range(num_frames)]
        for key in info:
            if key not in infos:
                infos[key] = []
            if type(info[key]) is list:
                infos[key].extend(info[key])
            else:
                infos[key].extend([info[key]] * num_frames)

        # eval_logger.info(
        #     f"{start_t:.2f}-{end_t:.2f}s | {hand_reference} | {prim} | {info['held_object']} | {info['idle_info']} | {info['held_obj_info']} | {info['interaction_info']}"
        # )
        print(
            f"{start_t:.2f}-{end_t:.2f}s | {prim} \n"
            f"\t Cropper info: {fast_mvt} | {hand_reference} ||| IP info: {info['idle_info']}\n"
            f"\t GRP info: {info['held_object']} | {info['held_obj_info']}"
            # f"\t TS info: {info['interaction_info']}\n"
        )

        # -------- Postprocessing -------

    idles: List[bool] = infos['idle']
    contacts: List[bool] = infos['contact']
    interactions: List[InteractionType] = infos['interaction']
    assert len(infos["raw_prims"]) == len(idles) == len(contacts) == len(interactions)
    N = len(infos["raw_prims"])

    # Smooth lists
    def _smooth_list(lst: List, chunk_size: int) -> List:
        for i in range(chunk_size, N - chunk_size, chunk_size):
            prev_window = lst[i - chunk_size]
            next_window = lst[i + chunk_size]
            if (prev_window == next_window) and (lst[i] != prev_window):
                for j in range(i, i + chunk_size):
                    lst[j] = prev_window
    _smooth_list(idles, chunk_max_frames)
    _smooth_list(contacts, chunk_max_frames)
    _smooth_list(interactions, chunk_max_frames)

    # 1) Start w/ empty list. First, fill in idle (priority). Then, the contact
    #    classes (transport, stabilize) with `interaction.`
    #    At the end of this step, prims contains "idle", "transport", "stabilize", and "UNK".
    prims = ["UNK"] * N
    prims = [HandPrimitives.IDLE if idle else prim for idle, prim in zip(idles, prims)]
    prims = [
        interaction if prim == "UNK" and contact else prim
        for interaction, contact, prim in zip(interactions, contacts, prims)
    ]

    # 2) Determine between reach, reposition, and reposition-reach
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
    # The list is now "idle", "reach", "reposition", "transport", "stabilize", and "UNK"
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
    side = chunk_max_frames // 2
    for i in range(1, N):
        if prims[i-1] == HandPrimitives.IDLE and prims[i] == HandPrimitives.TRANSPORT:
            for j in range(max(0, i-side), min(N, i+side)):
                prims[j] = HandPrimitives.REACH

        if prims[i-1] == HandPrimitives.TRANSPORT and prims[i] == HandPrimitives.IDLE:
            for j in range(max(0, i-side), min(N, i+side)):
                prims[j] = HandPrimitives.REPOSITION

    return prims, times, infos
