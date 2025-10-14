from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, asdict
import json
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

from lmms_eval.models.model_utils.load_video import load_long_video_decord
from tools.ultralytics_pose import Pose2DStream
from data.utils_strokerehab import PrimitiveLabelUtils

####################################### DATA CLASSES #######################################


@dataclass
class VideoChunk:
    should_infer: bool  # whether the VLM should be queried on this chunk, otherwise returns "N/A"
    moving_tracklet: bool # whether the hand crop is moving across frames
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    bboxes: List[List[int, int, int, int]]  # per-frame crop boxes
    start_t: float  # start time in seconds
    end_t: float    # end time in seconds

    gt_idle: bool     # ground truth idle state of the previous chunk
    gt_contact: bool   # ground truth contact state of the previous chunk


@dataclass
class HandKeypoints:
    wrist: List[np.ndarray]  # List of (x, y, conf) np.ndarrays for a video chnk
    elbow: List[np.ndarray]
    hand:  List[np.ndarray]


@dataclass
class LocatorOutput:
    left: HandKeypoints
    right: HandKeypoints


@dataclass
class CropperFromPoseOutput:
    should_infer: bool  # whether the VLM should be queried on this chunk, otherwise returns "N/A"
    moving_tracklet: bool  # whether the hand crop is moving across frames
    other_hand_in_view: bool  # whether the other hand is ever in view of any crop box
    bboxes: List[List[int, int, int, int]]  # per-frame crop boxes


@dataclass
class HandCtx:
    """
    The hand state that moves through the nodes.
    """
    handedness: str  # "left" | "right" the hand to track
    prev_pred_idle: bool = True
    prev_pred_contact: bool = False
    prev_gt_idle: bool = True
    prev_gt_contact: bool = False
    # Frames and hand location of the previous chunk.
    prev_frames: np.ndarray = field(default_factory=lambda: np.zeros((0, 224, 224, 3), dtype=np.uint8))
    prev_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class NodeOutput:
    output: str | bool
    info: Dict[str, Any]


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
        vlm: Optional[VLMProtocol],
        *,
        hand_wrist_elbow_ratio: float = 0.7  # 0.5 is normal for hand length to forearm length (do 0.7 to see a bit further)
    ):
        """
        Args:
            stream: Pose2DStream instance for 2D pose tracking.
            vlm: VQA model instance for person detection prompts.
            hand_wrist_elbow_ratio: Ratio of hand-to-wrist distance to wrist-to-elbow distance.
        """
        self.stream = stream
        self.vlm = vlm
        self.hand_wrist_elbow_ratio = hand_wrist_elbow_ratio
        self._person_detected = False
    
    def clear(self) -> None:
        self.stream.clear(keep_slot_labels=False, keep_pending_prompts=False)
        self.vlm.clear()
        self._person_detected = False

    def _handedness_to_idx(self, handedness: str) -> Tuple[int, int]:
        handedness = handedness.lower()
        assert handedness in ("left", "right")
        K2I = self.stream.KEYPOINT_TO_IDX
        if handedness == "right":
            return K2I["right_elbow"], K2I["right_wrist"]
        else:
            return K2I["left_elbow"], K2I["left_wrist"]

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
    ) -> LocatorOutput:
        """
        Returns: LocatorOutput
        """
        frames = np.asarray(frames)
        assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T, H, W, 3) RGB"
        T, H, W, _ = frames.shape

        # Ensure we are tracking the correct person (same as before).
        self.stream.reset_results()
        if not getattr(self, "_person_detected", False):
            self._person_detected = True
            prompt = person_locating_prompt if person_locating_prompt is not None else self.LOCATE_PROMPT
            patient_loc_text = self.vlm.process_frames([frames[0]], prompt)
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
        return LocatorOutput(
            left=HandKeypoints(**out["left"]),
            right=HandKeypoints(**out["right"]),
        )


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
) -> List[int, int, int, int]:
    half = side // 2
    cx_i = int(round(max(half, min(W - half, center[0]))))
    cy_i = int(round(max(half, min(H - half, center[1]))))
    x1, y1 = int(cx_i - half), int(cy_i - half)
    x2, y2 = int(cx_i + half), int(cy_i + half)
    # clamp to image bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    return [x1, y1, x2, y2]


class HandCropper:
    """
    Hand-centric cropping using 'hand' coordinates from HandLocator.
    """
    def __init__(
        self,
        *,
        use_cropping_strategy: bool = True,
        kp_conf_thresh: float = 0.9,
        other_hand_in_view_conf_thresh: float = 0.5,  # not used
        moving_tracklet_thresh: int = 12,
    ):
        """
        Hand-centric crop generator built on top of HandLocator.

        Strategy:
        - Uses the 'hand' keypoint from HandLocator (computed from wrist-elbow vector).
        - If cropping disabled, returns full-frame boxes but still checks if the other hand
            enters view.
        - Otherwise:
            X: every hand keypoint has high confidence
            Y: L-inf distance between endpoints >= moving_tracklet_thresh * (T - 1)
            • If X & Y -> moving crop (interpolated between start/end)
            • If X & not Y -> still crop (fixed at midpoint)
            • If not X -> no crop (full-frame)

        Args:
            use_cropping_strategy: If False, always return full-frame crops.
            kp_conf_thresh: Confidence threshold for hand keypoints.
            moving_tracklet_thresh: Movement threshold in pixels per frame for deciding “moving”.
            other_hand_in_view_conf_thresh: Confidence threshold for the other hand being in view.
        """
        self.use_cropping_strategy = bool(use_cropping_strategy)
        self.kp_conf_thresh = float(kp_conf_thresh)
        self.moving_tracklet_thresh = int(moving_tracklet_thresh)
        self.other_hand_in_view_conf_thresh = float(other_hand_in_view_conf_thresh)

    def clear(self) -> None:
        pass

    def process_frames(
        self,
        frames: np.ndarray,
        handedness: str,
        kps: LocatorOutput,
        *,
        bbox_side: int = 224,
    ) -> CropperFromPoseOutput:
        """
        Compute hand-centered crops for a clip.

        Returns: CropperFromPoseOutput
        """
        T, H, W, _ = frames.shape
        if not self.use_cropping_strategy:
            full = [0, 0, int(W), int(H)]
            crop_boxes = [full for _ in range(T)]
            should_infer, moving_tracklet = True, False
        else:
            # Try to crop if confident. Choices: moving crop, still crop, and no crop
            if handedness == "left":
                hand_kps, other_kps = (kps.left.hand, kps.right.hand)
            else:
                hand_kps, other_kps = (kps.right.hand, kps.left.hand)

            cur_first = hand_kps[0]
            cur_last = hand_kps[-1]
            cur_confs = [kp[2] for kp in hand_kps]

            # --- Step 1: Crop computation ---
            if not all(conf >= self.kp_conf_thresh for conf in cur_confs):
                full = [0, 0, int(W), int(H)]
                crop_boxes = [full for _ in range(T)]
                should_infer, moving_tracklet = False, False
            else:
                dist = max(abs(cur_last[0] - cur_first[0]), abs(cur_last[1] - cur_first[1]))
                fast_mvt = (dist >= self.moving_tracklet_thresh * max(1, (T - 1)))
                if fast_mvt:  # Do a moving crop if movement is fast
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
                    mid = ((cur_first[0] + cur_last[0]) / 2.0, (cur_first[1] + cur_last[1]) / 2.0)
                    box = _bbox_at_center_with_side(mid, side=bbox_side, W=W, H=H)
                    crop_boxes = [box for _ in range(T)]
                should_infer, moving_tracklet = True, fast_mvt

        # --- Step 2: Is the other hand ever in view? ---
        other_hand_in_view = False
        for hand_kp, crop_box in zip(other_kps, crop_boxes):
            if (
                hand_kp[2] >= self.other_hand_in_view_conf_thresh and \
                _bbox_contains_points(crop_box, ((hand_kp[0], hand_kp[1]),))
            ):
                other_hand_in_view = True
                break

        return CropperFromPoseOutput(
            should_infer, moving_tracklet, other_hand_in_view, crop_boxes
        )


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
        should_infer: bool,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        prompt: str,
    ) -> str:
        """
        Query the VLM with cropped frames whose size is dependent on the
        resolution and a formatted prompt.
        """
        if should_infer:
            cropped_frames = _get_cropped(frames=frames, bboxes=bboxes)
            return self.vlm.process_frames(cropped_frames, prompt)
        else:
            return "N/A"

    def run(
        self,
        should_infer: bool,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> NodeOutput:
        """
        Execute this node:
        Returns: (cur_state, info) where cur_state is this node's decision/output
            and info is a dictionary of related decision-making information.
        """
        ...


IDLE_PROMPT_METHODS = ("SMC", "Idle", "StatefulIdleFromPred", "StatefulIdleFromGT", "Focus")

class IdleProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on *HandCtx* and returns the next idle state
    and info related to the decision.
    """
    SMC_IDLE_PROMPT = (
        "Focus on the patient's {handedness} hand. Is it actively moving an object, "
        "moving towards an object, or moving away from an object? Answer YES or NO.\n\n"
    )
    IDLE_PROMPT = (
        "Is the patient's {handedness} hand idle? Answer 'Yes.' or 'No.' directly. \n"
    )
    STATEFUL_IDLE_PROMPT = (
        "This clip follows one where the patient's {handedness} hand was {prev_idle_str}. "
        "Answer directly: 'Active' if the hand remains active in this chunk; "
        "answer 'Idle' otherwise."
    )
    FOCUS_PROMPT = (
        "Focus ONLY on the patient's {handedness} hand. Completely ignore the other hand - "
        "pretend it does not exist. Even if the other hand is moving, touching objects, or idle, "
        "that information is irrelevant. Do not mention or consider the other hand in any way. "
        "Based only on the movement and posture of the patient's {handedness} hand, is it idle? "
        "Answer 'Yes.' or 'No.' directly."
    )

    def __init__(self, prompt_method: str, ctx: HandCtx, vlm: VLMProtocol, **fmt):
        assert prompt_method in IDLE_PROMPT_METHODS
        if prompt_method == "SMC":
            self.prompt = self.SMC_IDLE_PROMPT
        elif "StatefulIdle" in prompt_method:
            self.prompt = self.STATEFUL_IDLE_PROMPT
        elif prompt_method == "Focus":
            self.prompt = self.FOCUS_PROMPT
        else:
            self.prompt = self.IDLE_PROMPT
        self.prompt_method = prompt_method
        super().__init__(ctx, vlm, **fmt)

    def run(
        self,
        should_infer: bool,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> NodeOutput:
        """
        Execute this node:
        Returns: (cur_idle, info)
        """
        if self.prompt_method == "StatefulIdleFromGT":
            prev_idle = self.ctx.prev_gt_idle
        else:
            prev_idle = self.ctx.prev_pred_idle
        prev_idle_str = 'IDLE' if prev_idle else 'ACTIVE'
        fmt = {**self.fmt, "handedness": self.ctx.handedness, "prev_idle_str": prev_idle_str}
        prompt = self.prompt.format(**fmt)

        ans = self._query_vlm(should_infer, frames, bboxes, prompt)

        if self.prompt_method == "SMC":
            cur_idle = not ("yes" in ans.lower())
        elif self.prompt_method == "Idle":
            cur_idle = "yes" in ans.lower()
        elif self.prompt_method == "Focus":
            cur_idle = "yes" in ans.lower()
        else:
            cur_idle = "idle" in ans.lower()

        return NodeOutput(output=cur_idle, info={"prompt": prompt, "raw_answer": ans})


####################################### ORCHESTRATION SECTION #######################################


class HandStateMachine:
    def __init__(self,
                 *,
                 handedness: str,
                 idle_prompt_methods: Iterable[str] = IDLE_PROMPT_METHODS,
                 contact_prompt_methods: Iterable[str] = (),  # TODO
                 vlm: VLMProtocol,
                 target_objects: str = ""
    ):
        self.ctx = HandCtx(handedness=handedness)
        self.target_objects = target_objects

        self.ipns: Dict[str, IdleProcessingNode] = {}
        for idle_method in idle_prompt_methods:
            self.ipns[idle_method] = IdleProcessingNode(
                prompt_method=idle_method,
                ctx=self.ctx,
                vlm=vlm,
                target_objects=target_objects,
            )
        
        # TODO contact_prompt_methods

    def step(
        self, chunk: VideoChunk, include_prompt_info: bool = True
    ) -> Dict[str, str | bool]:
        """Processing order matters."""

        # Run VLM processing nodes
        idles: Dict[str, NodeOutput] = {}
        for method, ipn in self.ipns.items():
            idles[method] = ipn.run(
                chunk.should_infer,
                chunk.frames,
                chunk.bboxes,
            )
        # TODO contact

        # Update state
        self.ctx.prev_pred_idle = idles["StatefulIdleFromPred"].output
        self.ctx.prev_pred_contact = None  # TODO
        self.ctx.prev_gt_idle = chunk.gt_idle
        self.ctx.prev_gt_contact = chunk.gt_contact
        self.ctx.prev_frames = chunk.frames
        self.ctx.prev_bboxes = chunk.bboxes

        info = {
            **{idle_method: idles[idle_method].output for idle_method in IDLE_PROMPT_METHODS},
        }

        if include_prompt_info:
            info.update({
                idle_method + "_info": str(idles[idle_method].info) for idle_method in IDLE_PROMPT_METHODS
            })
        return info

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
    label_path: str,
    handedness: str,
    vlm: VLMProtocol,
    pose_stream: Pose2DStream,
    chunk_max_frames: int = 4,
    overlap_frames_num: int = 0,
    sampling_fps: int = 15,
    include_prompt_info_in_df: bool = False,
    idle_prompt_methods: Iterable[str] = IDLE_PROMPT_METHODS,
    contact_prompt_methods: Iterable[str] = (),
    crop_methods: Iterable[str] = ("window", "tracklet"),
) -> pd.DataFrame:
    """
    Arguments:
        video_path: path to the video file
        label_path: path to the label file
        handedness: "left" | "right", which hand to track
        vlm: the vision-language model that implements VLMProtocol
        pose_stream: the 2D pose predictor
        chunk_max_frames: number of frames per chunk
        sampling_strategy: "dense" | "uniform"
        overlap_frames_num: number of overlapping frames between chunks (only for "dense"). Keep at 0.
        sampling_fps: fps to sample the video at
    
    Returns a DataFrame with bool or str columns. Among other signals, this contains:
        'start_t': start time of a window
        'end_t': end time of a window
    """
    # NOTE: `gt_times` is longer than `gt_prims` by 1 element.
    # Its last element is the end time of the last primitive.
    gt_prims, gt_times = PrimitiveLabelUtils.convert_labels_to_prims_times(label_path)
    gt_prims_idx = 0

    # Always start with hand locator
    hand_locator = HandLocator(pose_stream, vlm)
    hand_locator.clear()

    # Two paths: window and tracklet
    target_objects = _get_target_objects(video_path)

    visual_pathways: Dict[str, Tuple[HandCropper, HandStateMachine]] = {
        "window": (
            HandCropper(use_cropping_strategy=False),
            HandStateMachine(
                handedness=handedness,
                vlm=vlm,
                target_objects=target_objects,
                idle_prompt_methods=idle_prompt_methods,
                contact_prompt_methods=contact_prompt_methods)
        ),
        "tracklet": (
            HandCropper(),
            HandStateMachine(
                handedness=handedness,
                vlm=vlm,
                target_objects=target_objects,
                idle_prompt_methods=idle_prompt_methods,
                contact_prompt_methods=contact_prompt_methods)
        )
    }
    if "window" not in crop_methods:
        visual_pathways.pop("window")
    if "tracklet" not in crop_methods:
        visual_pathways.pop("tracklet")

    infos = {}

    for frames, start_t, end_t in load_long_video_decord(
        video_path,
        max_frames_num=chunk_max_frames,
        sampling_strategy="dense",
        overlap_frames_num=overlap_frames_num,
        sampling_fps=sampling_fps,
        force_sample=False,
        ret_idx=False,
    ):
        info = {}
        info["start_t"] = start_t
        info["end_t"] = end_t

        # Zoom to the last primitive label before start_t
        while (gt_prims_idx + 1 < len(gt_prims)) and (gt_times[gt_prims_idx + 1] < start_t):
            gt_prims_idx += 1
        gt_idle_for_prompt = (gt_prims[gt_prims_idx] == "idle")
        gt_contact_for_prompt = (gt_prims[gt_prims_idx] in ("transport", "stabilize"))

        # Extract hand keypoints
        locator_output = hand_locator.process_frames(frames)
        info.update({
            "left_wrist_kps": [list(kp) for kp in locator_output.left.wrist],
            "left_elbow_kps": [list(kp) for kp in locator_output.left.elbow],
            "left_hand_kps":  [list(kp) for kp in locator_output.left.hand],
            "right_wrist_kps": [list(kp) for kp in locator_output.right.wrist],
            "right_elbow_kps": [list(kp) for kp in locator_output.right.elbow],
            "right_hand_kps":  [list(kp) for kp in locator_output.right.hand],
        })

        # Two different visual prompt pathways
        for lbl, (hand_cropper, machine) in visual_pathways.items():
            cropper_output = hand_cropper.process_frames(
                frames, handedness, locator_output, bbox_side=224
            )
            for key, value in asdict(cropper_output).items():
                info[f"{lbl}_{key}"] = value

            chunk = VideoChunk(
                should_infer=cropper_output.should_infer,
                moving_tracklet=cropper_output.moving_tracklet,
                frames=frames,
                bboxes=cropper_output.bboxes,
                start_t=start_t,
                end_t=end_t,
                gt_idle=gt_idle_for_prompt,
                gt_contact=gt_contact_for_prompt,
            )
            vlm_info = machine.step(chunk, include_prompt_info=include_prompt_info_in_df)
            for key, value in vlm_info.items():
                info[f"{lbl}_{key}"] = value

        for key in info:
            if key not in infos:
                infos[key] = []
            infos[key].append(info[key])

        print(f"Processed {video_path} | {start_t:.2f}-{end_t:.2f}s ...")

    df = pd.DataFrame(infos)
    return df
