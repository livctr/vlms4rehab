"""
Conditional Prompting and Cropping Logic specifically designed for
RTT and shelf tasks. Custom prompts and post-processing.
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
    """Pose model and cropping status."""
    ABSTAIN       = "ABSTAIN"       # ABSTAIN due to low confidence
    OK_NO_MVT     = "OK [NO MVT]"   # OK, minimal motion
    OK            = "OK"            # OK, some motion
    FAST_MOVEMENT = "FAST MOVEMENT" # FAST MOVEMENT detected


class IdleState:
    """Idle vs. all."""
    IDLE          = "idle"
    ACTIVE        = "active"


class GraspState:
    """Transport/stabilize v. reach/reposition."""
    EMPTY         = "empty"
    HOLDING       = "holding"


class InteractionState:
    """Transport v. stabilize. (reach v. reposition are figured out in post-processing)."""
    TRANSPORT     = "transport"
    STABILIZE     = "stabilize"


@dataclass
class VideoChunk:
    pose_status: str
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
    pose_status: str        = HandStateStatus.OK
    idle_status: str        = IdleState.IDLE
    grasp_status: str       = GraspState.EMPTY
    interaction_status: str = InteractionState.TRANSPORT

    frames: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


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
        # 0.5 is normal for hand length to forearm length (do 0.7 to see a bit further)
        hand_wrist_elbow_ratio: float = 0.7
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
            patient_loc_text = self.vqa_model.process_frames(frames[0], prompt)
            try:
                bbox, label = _extract_largest_bbox_and_label(patient_loc_text)
            except (ValueError, KeyError, json.JSONDecodeError):
                eval_logger.warning(
                    f"Patient localization failed (response: {patient_loc_text!r}). "
                    f"Falling back to full-frame bbox."
                )
                bbox, label = (0.0, 0.0, float(W), float(H)), None
            self.stream.add_new_person_to_track(bbox=bbox, label=label)

        # Ensure we get the right elbow/wrist keypoints
        kp_elbow, kp_wrist = self._handedness_to_idx(handedness)

        kps_wrist, kps_elbow, kps_hand = [], [], []
        for i in range(T):
            kps = self.stream.process_frame(frames[i])  # (1, num_person, 17, 3)
            kp = kps[0, 0]

            wx, wy, wc = kp[kp_wrist]
            ex, ey, ec = kp[kp_elbow]

            # check for NaNs
            if (
                np.isnan(wx) or np.isnan(wy) or np.isnan(wc)
                or np.isnan(ex) or np.isnan(ey) or np.isnan(ec)
            ):
                placeholder = np.array([np.nan, np.nan, 0.0], dtype=kp.dtype)
                kps_wrist.append(placeholder)
                kps_elbow.append(placeholder)
                kps_hand.append(placeholder)
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

            kps_wrist.append(np.array([wx, wy, float(wc)], dtype=kp.dtype))
            kps_elbow.append(np.array([ex, ey, float(ec)], dtype=kp.dtype))
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
        fast_movement_thresh: int = 15,
        min_frames_for_movement: int = 4,
        is_moving_thresh: int = 3
    ):
        """
        Args:
            kp_conf_thresh: Minimum confidence for wrist & elbow keypoints to consider valid.
            fast_movement_thresh: If the L-inf distance between the first and last frame centers
                                    exceeds this threshold times the number of frames minus one,
                                    the status is "FAST MOVEMENT".
        """
        self.kp_conf_thresh = float(kp_conf_thresh)
        self.fast_movement_thresh = int(fast_movement_thresh)
        self.min_frames_for_movement = int(min_frames_for_movement)
        self.is_moving_thresh = int(is_moving_thresh)
        self._last_center: Optional[Tuple[float, float]] = None
        self._detected = False

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
        first = hand_kps[0]
        last = hand_kps[-1]
        ok = (first[2] >= self.kp_conf_thresh) and (last[2] >= self.kp_conf_thresh)
        if not ok:
            full = (0, 0, int(W), int(H))
            return "ABSTAIN", [full for _ in hand_kps]
    
        dist = max(abs(last[0] - first[0]), abs(last[1] - first[1]))
        slow_mvt = (dist >= self.is_moving_thresh * (T - 1)) and (T >= self.min_frames_for_movement)
        fast_mvt = (dist >= self.fast_movement_thresh * (T - 1)) and (T >= self.min_frames_for_movement)
    
        if fast_mvt:
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

        else:
            crop_box = self._bbox_at_center_with_side(
                ((first[0] + last[0]) / 2.0, (first[1] + last[1]) / 2.0),
                side=bbox_side, W=W, H=H
            )
            crop_boxes = [crop_box for _ in range(T)]

        pose_status = HandStateStatus.FAST_MOVEMENT if fast_mvt else (
            HandStateStatus.OK if slow_mvt else HandStateStatus.OK_NO_MVT
        )
        return pose_status, crop_boxes

        


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

    def __init__(self, ctx: HandCtx, vlm: VLMProtocol, handedness: str, do_crop: bool = True):
        self.ctx = ctx
        self.vlm = vlm
        self.handedness = handedness
        if do_crop:
            self.fmt = {"the_hand": "the hand"}
        else:
            self.fmt = {"the_hand": f"the patient's {handedness} hand"}

        # do_crop should curate the input frames (by cropping), the conditionals within each
        # node, and the prompts themselves accordingly.
        self.do_crop = do_crop

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
        if self.do_crop:
            frames = _get_cropped(frames=orig_frames, bboxes=bboxes)
        else:
            frames = orig_frames
        return self.vlm.process_frames(frames, prompt)

    def _run(
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
        if not self.do_crop and pose_status in (HandStateStatus.ABSTAIN, HandStateStatus.FAST_MOVEMENT):
            pose_status = HandStateStatus.OK  # ignore pose status if not cropping
        return self._run(pose_status, frames, bboxes)


class MotionProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on* HandContactCtx and returns the next motion
    state.
    """
    IDLE_PROMPT = (
        "Is {the_hand} idle in this video clip? \n"
        "(Idle) Visibly resting on the black mat, not moving, and not interacting with a cylindrical block. "
        "(Active) In the air, moving towards an object, moving away from an object, interacting with a "
        "cylindrical block, or 'resting' on a cylindrical block. The hand can be moving very slowly through the air and still "
        "be considered 'active.'"
        "Answer 'Yes.' if {the_hand} is idle; answer 'No.' otherwise.\n"
    )

    def _run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        if pose_status == HandStateStatus.ABSTAIN:
            return IdleState.IDLE, {"method": "MPN [abstain]"}
        elif pose_status == HandStateStatus.FAST_MOVEMENT:
            return IdleState.ACTIVE, {"method": "MPN [fast movement]"} 
        else:
            ans = self._query_vlm(frames, bboxes, self.IDLE_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return IdleState.IDLE, {"method": "MPN", "result": f"Ans: {ans}"}
            else:
                return IdleState.ACTIVE, {"method": "MPN", "result": f"Ans: {ans}"}


class GraspProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on* HandContactCtx and returns the next contact
    state.
    """
    GRASP_PROMPT = (
        "This is a chunk from a video sequence. In the previous chunk, {the_hand} was not holding anything. "
        "Answer directly: 'Yes.' if {the_hand} visibly picks up a cylindrical block in this chunk; answer 'No.' otherwise.\n"
        "Mere contact does not count as grasping.\n"
    )
    RELEASE_PROMPT = (
        "This is a chunk from a video sequence. In the previous chunk, {the_hand} was holding a cylindrical block. "
        "Answer directly: 'Yes.' if {the_hand} puts down and releases the block; "
        "answer 'No.' otherwise.\n"
    )
    
    def _run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        if pose_status == HandStateStatus.ABSTAIN:
            return GraspState.EMPTY, {"method": "GRP [abstain]"}
        if pose_status == HandStateStatus.FAST_MOVEMENT:
            return self.ctx.grasp_status, {"method": "GRP [abstain]"}

        if self.ctx.grasp_status == GraspState.HOLDING:
            ans = self._query_vlm(frames, bboxes, self.RELEASE_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return GraspState.EMPTY, {"method": "GRP", "result": f"Ans: {ans}"}
            else:
                return GraspState.HOLDING, {"method": "GRP", "result": f"Ans: {ans}"}
        else:
            ans = self._query_vlm(frames, bboxes, self.GRASP_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return GraspState.HOLDING, {"method": "GRP", "result": f"Ans: {ans}"}
            else:
                return GraspState.EMPTY, {"method": "GRP", "result": f"Ans: {ans}"}


class InteractionProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on *HandCtx* and returns the next interaction state
    and info related to the decision.
    """
    def _run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_idle, info)
        """
        if pose_status == HandStateStatus.OK_NO_MVT:
            return InteractionState.STABILIZE, {"method": "TS [no movement]"}
        else:
            return InteractionState.TRANSPORT, {"method": "TS [default]"}


####################################### ORCHESTRATION SECTION #######################################


class HandStateMachine:
    """
    Orchestrates the conditional prompting logic.
    """
    def __init__(self, vlm: VLMProtocol, handedness: str, do_crop: bool = True):
        self.ctx = HandCtx(handedness=handedness)
        self.motion_node = MotionProcessingNode(self.ctx, vlm, handedness=handedness, do_crop=do_crop)
        self.grasp_node = GraspProcessingNode(self.ctx, vlm, handedness=handedness, do_crop=do_crop)
        self.i_node = InteractionProcessingNode(self.ctx, vlm, handedness=handedness, do_crop=do_crop)

    def step(
        self, chunk: VideoChunk
    ) -> Tuple[str, Dict[str, Any]]:

        frames = chunk.frames
        bboxes = chunk.bboxes

        idle, idle_info = self.motion_node.run(
            chunk.pose_status, frames, bboxes
        )
        self.ctx.idle_status = idle

        grasp, grasp_info = self.grasp_node.run(
            chunk.pose_status, frames, bboxes
        )
        self.ctx.grasp_status = grasp

        interaction, interaction_info = self.i_node.run(
            chunk.pose_status, frames, bboxes
        )
        self.ctx.interaction_status = interaction

        self.ctx.pose_status = chunk.pose_status

        if idle == IdleState.IDLE:
            prim = "idle"
        else:
            if grasp == GraspState.EMPTY:
                prim = "reach"
            else:
                if interaction == InteractionState.TRANSPORT:
                    prim = "transport"
                else:
                    prim = "stabilize"

        info = {
            "idle": idle, "idle_info": idle_info,
            "grasp": grasp, "grasp_info": grasp_info,
            "interaction": interaction, "interaction_info": interaction_info,
        }
        return prim, info

    @property
    def context(self) -> HandCtx:
        return self.ctx


def expand(lst, element, w_size):
    """
    Fill a full window with `element` if both its neighboring full windows
    are entirely equal to `element`. Also extend a trailing partial window
    if the preceding full window is entirely `element`.
    """
    for w_start in range(w_size, len(lst) - w_size, w_size):
        if lst[w_start - w_size] == element and lst[w_start + w_size] == element:
            for idx in range(w_start, w_start + w_size):
                lst[idx] = element
        
    # Change last partial window if it doesn't match with the previous full window
    if len(lst) % w_size != 0:
        w_start = len(lst) - (len(lst) % w_size)
        if w_start - w_size >= 0 and lst[w_start - w_size] == element:
            for idx in range(w_start, len(lst)):
                lst[idx] = element


def predict_with_state_machine(
    video_path: str,
    handedness: str,
    vlm: VLMProtocol,
    pose_stream: Pose2DStream,
    max_frames_num: int = 4,
    sampling_strategy: str = "dense",
    overlap_frames_num: int = 0,
    sampling_fps: int = 15,
    do_crop: bool = True,
    do_postprocess: bool = True,
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
        do_crop: whether to use hand-centric cropping
        do_postprocess: whether to post-process the raw predictions

    Returns a list of primitives (one per frame), their timestamps, and detailed info.
    """
    machine = HandStateMachine(vlm, handedness, do_crop=do_crop)
    hand_locator = HandLocator(pose_stream, vlm, hand_wrist_elbow_ratio=0.75)
    hand_cropper = HandCropper()
    hand_locator.clear()
    hand_cropper.clear()

    infos = {}
    for frames, start_t, end_t in load_long_video_decord(
        video_path,
        max_frames_num=max_frames_num,
        sampling_strategy=sampling_strategy,
        overlap_frames_num=overlap_frames_num,
        sampling_fps=sampling_fps,
        force_sample=False,
        ret_idx=False,
    ):
        num_frames = len(frames)

        # Run the pose model always to get pose_status. Change boxes if not cropping.
        kp = hand_locator.process_frames(frames, handedness=handedness)
        kp_wrist, kp_elbow, kp_hand = kp["wrist"], kp["elbow"], kp["hand"]
        pose_status, boxes = hand_cropper.process_frames(frames, hand_kps=kp_hand, bbox_side=224)

        chunk = VideoChunk(
            pose_status=pose_status,
            frames=frames,
            bboxes=boxes,  # Can use or ignore
            start_t=start_t,
            end_t=end_t
        )
        prim, info = machine.step(chunk)

        # Infos
        info["times"] = np.linspace(start_t, end_t, num_frames, endpoint=False).tolist()
        info["pose_status"] = [pose_status] * num_frames
        info["raw_prims"] = [prim] * num_frames
        info["kps_wrist"] = [kp_wrist[i] for i in range(num_frames)]
        info["kps_elbow"] = [kp_elbow[i] for i in range(num_frames)]
        info["kps_hand"] = [kp_hand[i] for i in range(num_frames)]
        info["bboxes"] = [boxes[i] for i in range(num_frames)]
        for key in info:
            if key not in infos:
                infos[key] = []
            if type(info[key]) is list:
                infos[key].extend(info[key])
            else:
                infos[key].extend([info[key]] * num_frames)
        
        eval_logger.info(
            f"{start_t:.2f}-{end_t:.2f}s | Pose: {pose_status} | Ans: {prim}"
        )
        print(
            f"{start_t:.2f}-{end_t:.2f}s | Pose: {pose_status} | Ans: {prim} | Idle: {info['idle_info']}"
        )

    raw_prims = infos.pop("raw_prims", [])
    times = infos.pop("times", [])
    if len(raw_prims) == 0 or len(times) == 0:
        return [], [], {}

    # Do simple post-processing (convert "reach" to "reposition" if ends with non-grasp primitive)
    if not do_postprocess:
        T = len(raw_prims)
        grasp_ahead = False  # No "grasp" at the end
        for i in range(T-1, -1, -1):
            if raw_prims[i] == "transport" or raw_prims[i] == "stabilize":
                grasp_ahead = True
            elif raw_prims[i] == "idle":
                grasp_ahead = False
            elif raw_prims[i] == "reach":
                if not grasp_ahead:
                    raw_prims[i] = "reposition"
        return raw_prims, times, infos

    idles = infos.get("idle", [])
    grasps = infos.get("grasp", [])
    interactions = infos.get("interaction", [])
    assert len(raw_prims) == len(idles) == len(grasps) == len(interactions) == len(times)

    ### Post-process the signals
    T = len(raw_prims)

    # Dilate and erode to smooth idles and grasps
    expand(idles, IdleState.ACTIVE, max_frames_num)
    expand(idles, IdleState.IDLE, max_frames_num)
    for i in range(T):
        grasps[i] = GraspState.EMPTY if idles[i] == IdleState.IDLE else grasps[i]
    expand(grasps, GraspState.HOLDING, max_frames_num)
    expand(grasps, GraspState.EMPTY, max_frames_num)

    prims = ["UNK" for _ in range(T)]
    # Fill in idle and transport (transport as a place-holder for transport+stabilize)
    prims = ["idle" if idles[i] == IdleState.IDLE else prims[i] for i in range(T)]
    prims = ["transport" if (prims[i] == "UNK" and grasps[i] == GraspState.HOLDING) else prims[i] for i in range(T)]

    # Fill UNK with reach/reposition depending on past and future knowns
    active_unk = False
    unk_left = "border"
    unk_start = -1    
    for i in range(T+1):
        if i < T and prims[i] == "UNK":
            if not active_unk:
                unk_start = i
                active_unk = True
        else:
            if active_unk:
                unk_end = "border" if i == T else prims[i]
                # 9 possibilities
                if unk_left == "border" and unk_end == "border":
                    # Don't expect this case
                    for j in range(unk_start, i):
                        prims[j] = "idle"
                elif unk_left == "idle" and unk_end == "idle":
                    mid = (unk_start + i - 1) // 2
                    for j in range(unk_start, mid + 1):
                        prims[j] = "reach"
                    for j in range(mid + 1, i):
                        prims[j] = "reposition"
                elif unk_left == "transport" and unk_end == "transport":
                    mid = (unk_start + i - 1) // 2
                    for j in range(unk_start, mid + 1):
                        prims[j] = "reposition"
                    for j in range(mid + 1, i):
                        prims[j] = "reach"
                elif unk_end == "transport":
                    for j in range(unk_start, i):
                        prims[j] = "reach"
                else:
                    for j in range(unk_start, i):
                        prims[j] = "reposition"
                active_unk = False
            unk_left = "border" if i == T else prims[i]

    # Smooth out quick idles -> contacts and contacts -> idles 
    side = max_frames_num // 2
    for i in range(1, T):
        if prims[i-1] == "idle" and prims[i] == "transport":
            for j in range(max(0, i-side), min(T, i+side)):
                prims[j] = "reach"

        if prims[i-1] == "transport" and prims[i] == "idle":
            for j in range(max(0, i-side), min(T, i+side)):
                prims[j] = "reposition"
    
    # Integrate stabilize
    active_grasp = False
    grasp_start = -1
    for i in range(T+1):
        if i < T and grasps[i] == GraspState.HOLDING:
            if not active_grasp:
                grasp_start = i
                active_grasp = True
        else:
            if not active_grasp:
                continue
            active_grasp = False

            grasp_end = i

            stabilize_segments = []
            active_stabilize = False
            stabilize_start = -1
            for j in range(grasp_start, grasp_end):
                if interactions[j] == InteractionState.STABILIZE:
                    if not active_stabilize:
                        stabilize_start = j
                        active_stabilize = True
                else:
                    if active_stabilize:
                        stabilize_segments.append((stabilize_start, j))
                        active_stabilize = False
            if active_stabilize:
                stabilize_segments.append((stabilize_start, grasp_end))

            # `stabilize_segments` is sorted by construction
            # Two cases where we actually integrate stabilize into prims
            # 1. Stabilize is long: at least 3*max_frames_num frames
            # 2. Stabilize is within 3 windows of the grasp_end -> convert last segment
            # 3. If we convert the last segment, smooth with previous segment if close enough
            last_active_s_end = -1
            for (s_start, s_end) in stabilize_segments:
                length = s_end - s_start
                if length >= 2 * max_frames_num:
                    for j in range(s_start, s_end):
                        prims[j] = "stabilize"
                    last_active_s_end = s_end
                elif grasp_end - s_end <= 3 * max_frames_num:
                    for j in range(grasp_end - max_frames_num, grasp_end):
                        prims[j] = "stabilize"
                    if (grasp_end - max_frames_num) - last_active_s_end < max_frames_num:
                        # Merge with previous segment
                        for j in range(last_active_s_end, grasp_end - max_frames_num):
                            prims[j] = "stabilize"

    return prims, times, infos
