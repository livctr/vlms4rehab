"""
Tests conditional prompting and cropping.

In `predict_with_state_machine`, set the `cropping`
and `conditional_prompting` flags to True/False.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from lmms_eval.models.model_utils.load_video import load_long_video_decord
from lmms_eval.tasks.strokerehab.utils_primitives import _convert_motion_contact_to_primitives
from tools.ultralytics_pose import Pose2DStream
from loguru import logger as eval_logger

####################################### DATA CLASSES #######################################


class HandStateStatus:
    OK            = "OK"
    ABSTAIN       = "ABSTAIN"
    FAST_MOVEMENT = "FAST MOVEMENT"


class MovingState:
    STATIONARY    = "stationary"
    MOVING        = "moving"


class GraspState:
    EMPTY         = "empty"
    HOLDING       = "holding"


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
    pose_status: str = HandStateStatus.OK  # "OK" | "ABSTAIN" | "FAST MOVEMENT"
    moving_status: str = MovingState.STATIONARY  # "stationary" | "moving"
    grasp_status: str = GraspState.EMPTY  # "empty" | "holding"


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
            patient_loc_text = self.vqa_model.process_frames(frames[0], prompt)
            bbox, label = _extract_largest_bbox_and_label(patient_loc_text)
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


class MotionProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on* HandContactCtx and returns the next motion
    state.
    """
    MOTION_PROMPT = (
        "Focus on the patient's {handedness} hand. Is it actively moving an object, moving towards "
        "an object, or moving away from an object? Answer YES or NO."
    )

    PREV_STILL_PROMPT = (
        "Focus on the patient's {handedness} hand. It was previously still. Is it now actively moving "
        "an object, moving towards an object, or moving away from an object? Answer YES or NO."
    )

    PREV_MOVING_PROMPT = (
        "Focus on the patient's {handedness} hand. It was previously moving an object or moving "
        "toward/away from one. Is it now still? Answer YES or NO."
    )

    def __init__(self, ctx: HandCtx, vlm: VLMProtocol, conditional: bool = True, **fmt):
        super().__init__(ctx, vlm, **fmt)
        self.conditional = conditional

    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.conditional:
            ans = self._query_vlm(frames, bboxes, self.MOTION_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return MovingState.MOVING, {"method": "MPN [uncond]", "result": f"Ans: {ans}"}
            else:
                return MovingState.STATIONARY, {"method": "MPN [uncond]", "result": f"Ans: {ans}"}
        # Conditional prompting
        if self.ctx.moving_status == MovingState.STATIONARY:
            ans = self._query_vlm(frames, bboxes, self.PREV_STILL_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return MovingState.MOVING, {"method": "MPN [prev still]", "result": f"Ans: {ans}"}
            else:
                return MovingState.STATIONARY, {"method": "MPN [prev still]", "result": f"Ans: {ans}"}
        else:
            ans = self._query_vlm(frames, bboxes, self.PREV_MOVING_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return MovingState.STATIONARY, {"method": "MPN [prev moving]", "result": f"Ans: {ans}"}
            else:
                return MovingState.MOVING, {"method": "MPN [prev moving]", "result": f"Ans: {ans}"}


class GraspProcessingNode(ProcessingNode):
    """
    Stateless node object that *operates on* HandContactCtx and returns the next contact
    state.
    """
    GRASP_PROMPT = (
        "Focus on the patient's {handedness} hand. Is it actively grasping or holding an object? Answer YES or NO."
    )

    PREV_GRASP_PROMPT = (
        "Focus on the patient's {handedness} hand. Previously, it was actively grasping an object. "
        "Does it release the object in this clip? Answer YES or NO directly."
    )

    PREV_EMPTY_PROMPT = (
        "Focus on the patient's {handedness} hand. Previously, the hand was empty. Does it "
        "grasp an object in this clip? Answer YES or NO directly."
    )

    def __init__(self, ctx: HandCtx, vlm: VLMProtocol, conditional: bool = True, **fmt):
        super().__init__(ctx, vlm, **fmt)
        self.conditional = conditional
    
    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.conditional:
            ans = self._query_vlm(frames, bboxes, self.GRASP_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return GraspState.HOLDING, {"method": "GRP [uncond]", "result": f"Ans: {ans}"}
            else:
                return GraspState.EMPTY, {"method": "GRP [uncond]", "result": f"Ans: {ans}"}
        # Conditional prompting
        if self.ctx.grasp_status == GraspState.EMPTY:
            ans = self._query_vlm(frames, bboxes, self.PREV_EMPTY_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return GraspState.HOLDING, {"method": "GRP [prev empty]", "result": f"Ans: {ans}"}
            else:
                return GraspState.EMPTY, {"method": "GRP [prev empty]", "result": f"Ans: {ans}"}
        else:
            ans = self._query_vlm(frames, bboxes, self.PREV_GRASP_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                return GraspState.EMPTY, {"method": "GRP [prev grasp]", "result": f"Ans: {ans}"}
            else:
                return GraspState.HOLDING, {"method": "GRP [prev grasp]", "result": f"Ans: {ans}"}


####################################### ORCHESTRATION SECTION #######################################


class HandStateMachine:
    """
    Orchestrates the conditional prompting logic.
    """
    def __init__(self, vlm: VLMProtocol, handedness: str, conditional: bool = True, **fmt):
        self.ctx = HandCtx(handedness=handedness)
        self.motion_node = MotionProcessingNode(self.ctx, vlm, conditional=conditional, handedness=handedness)
        self.grasp_node = GraspProcessingNode(self.ctx, vlm, conditional=conditional, handedness=handedness)

    def step(
        self, chunk: VideoChunk
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:

        motion, motion_info = self.motion_node.run(
            chunk.pose_status, chunk.frames, chunk.bboxes
        )
        self.ctx.moving_status = motion

        grasp, grasp_info = self.grasp_node.run(
            chunk.pose_status, chunk.frames, chunk.bboxes
        )
        self.ctx.grasp_status = grasp

        self.ctx.pose_status = chunk.pose_status

        moving = "Yes" if self.ctx.moving_status == MovingState.MOVING else "No"
        holding = "Yes" if self.ctx.grasp_status == GraspState.HOLDING else "No"
        s = f"{moving} <SEP> {holding}"
        info = {
            "motion_info": motion_info,
            "grasp_info": grasp_info,
        }
        return s, info

    @property
    def context(self) -> HandCtx:
        return self.ctx


def predict_with_state_machine(
    video_path: str,
    handedness: str,
    vlm: VLMProtocol,
    pose_stream: Pose2DStream,
    max_frames_num: int = 8,
    sampling_strategy: str = "dense",
    overlap_frames_num: int = 0,
    sampling_fps: int = 15,
    conditional_prompting: bool = True,
    cropping: bool = True,
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
        conditional_prompting: whether to use conditional prompting in the state machine
        cropping: whether to use hand-centric cropping

    Returns a list of primitives (one per frame), their timestamps, and detailed info.
    """
    machine = HandStateMachine(vlm, handedness, conditional=conditional_prompting)
    hand_locator = HandLocator(pose_stream, vlm)
    hand_cropper = HandCropper(
        kp_conf_thresh=0.90, fast_movement_thresh=15, interpolation="mixed"
    )
    hand_locator.clear()
    hand_cropper.clear()

    start_times, answers = [], []
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
        if not cropping:
            box = (0, 0, frames.shape[2], frames.shape[1])
            boxes = [box for _ in range(num_frames)]

        chunk = VideoChunk(
            pose_status=pose_status,
            frames=frames,
            bboxes=boxes,
            start_t=start_t,
            end_t=end_t
        )
        ans, info = machine.step(chunk)

        start_times.append(start_t)
        answers.append(ans)

        # Infos
        info["times"] = np.linspace(start_t, end_t, num_frames, endpoint=False).tolist()
        info["pose_status"] = [pose_status] * num_frames
        info["answer"] = [ans] * num_frames
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
            f"{start_t:.2f}-{end_t:.2f}s | Pose: {pose_status} | Ans: {ans}"
        )
        print(
            f"{start_t:.2f}-{end_t:.2f}s | Pose: {pose_status} | Ans: {ans}"
        )

    if len(answers) == 0:
        return [], [], {}

    start_times.append(end_t)  # for the conversion API
    # Ensure consistency with SMC processing
    primitives, _ = _convert_motion_contact_to_primitives(answers, start_times)
    return primitives, start_times, infos
