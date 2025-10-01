"""
State machine implementation for IDLE-MOTION-CONTACT detection.

Organization
- Data classes (HandContactCtx, HandIdleCtx, VideoChunk, VLMProtocol)
- IDLE section (prompts, nodes)
- CONTACT section (prompts, nodes)
- Orchestrator (the state machine and main prediction function)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type

import numpy as np

from lmms_eval.models.model_utils.load_video import load_long_video_decord
from tools.ultralytics_pose import Pose2DStream
from vic_pipe.contact_v2 import HandLocator, HandCropper

logging.basicConfig(
    filename="results/statemachine.log",  # your log file
    filemode="a",  # append mode
    level=logging.INFO,  # or DEBUG if you want more verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


####################################### DATA CLASSES #######################################


class HandStateStatus:
    OK = "OK"
    ABSTAIN = "ABSTAIN"
    FAST_MOVEMENT = "FAST MOVEMENT"


@dataclass
class HandContactCtx:
    """
    The hand state that moves through the nodes.
    """

    handedness: str  # "left" | "right"
    status: str = HandStateStatus.OK  # "OK" | "ABSTAIN" | "FAST MOVEMENT"
    # Whether in contact with an object (List in case multiple contacts in a window)
    contacts: List[bool] = field(default_factory=lambda: [False])
    last_contact_box: Optional[Tuple[int, int, int, int]] = None
    last_ok_box: Optional[Tuple[int, int, int, int]] = (
        None  # (x1, y1, x2, y2) of last frame in last chunk that is ok
    )
    held_object: Optional[str] = None


@dataclass
class HandIdleCtx:
    """
    The IDLE v. NOT IDLE state that moves through the nodes.
    """

    handedness: str  # "left" | "right"
    # Whether idle (List in case multiple contacts in a window). List length will be 1,
    # but just for consistency.
    idles: List[bool] = field(default_factory=lambda: [True])


@dataclass
class VideoChunk:
    pose_status: str
    frames: np.ndarray
    bboxes: List[Tuple[int, int, int, int]]
    start_t: float
    end_t: float


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


####################################### IDLE SECTION #######################################

##### --------------------------------- Prompts --------------------------------- #####
# IDLE_PROMPT = (
#     "Is the hand idle (i.e. resting on a table or surface without intent "
#     "to grasp or release any objects)? Answer 'Yes.' or 'No.' directly.\n"
# )
IDLE_PROMPT = (
    "Is the hand at rest on a surface without intent to grasp or release any objects? "
    # "Is the hand idle (not moving or barely moving)? Answer 'Yes.' or 'No.' directly.\n"
)

##### --------------------------------- Nodes --------------------------------- #####


def _get_cropped(
    frames: np.ndarray, bbox: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    cropped_frames = []
    for frame, (x1, y1, x2, y2) in zip(frames, bbox):
        cropped = frame[y1:y2, x1:x2]  # standard numpy slicing
        cropped_frames.append(cropped)
    return np.stack(cropped_frames, axis=0)


class IdleStateNode:
    """
    Stateless node object that *operates on* HandIdleCtx and returns the next node type
    (idle or not idle).
    """

    def __init__(self, idle_ctx: HandIdleCtx, vlm: VLMProtocol):
        self.idle_ctx = idle_ctx
        self.vlm = vlm

    def _query_vlm(self, frames: np.ndarray, prompt: str) -> str:
        return self.vlm.process_frames(frames, prompt)

    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[bool], Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_idle, info)
        """
        prev_idle = self.idle_ctx.idles[-1]
        if pose_status == HandStateStatus.ABSTAIN:
            # Use the previous state
            return [prev_idle], {
                "method": "repeat",
                "outputs": f"Idle: {prev_idle} -> {prev_idle}",
            }
        elif pose_status == HandStateStatus.FAST_MOVEMENT:
            # Definitely not idle
            return [False], {
                "method": "fast_movement",
                "outputs": f"Idle: {prev_idle} -> False",
            }
        elif pose_status == HandStateStatus.OK:
            # OK: ask the VLM
            cropped_frames = _get_cropped(frames, bboxes)  # Zoom in on the correct hand
            ans = self._query_vlm(cropped_frames, IDLE_PROMPT).lower()
            if "yes" in ans:
                return [True], {
                    "method": "prompt",
                    "outputs": f"Idle: {prev_idle} -> True. Ans: {ans}",
                }
            else:
                return [False], {
                    "method": "prompt",
                    "outputs": f"Idle: {prev_idle} -> False. Ans: {ans}",
                }
        else:
            raise ValueError(f"Unknown pose status: {pose_status}")


####################################### CONTACT SECTION #######################################

##### --------------------------------- Prompts --------------------------------- #####

BASIC_CONTACT_PROMPT = (
    "Is the hand holding something? Answer 'Yes.' or 'No.' directly.\n"
)
WHICH_OBJECT_IN_CONTACT_PROMPT = (
    "Target objects: {target_objects}\n\n"
    "Which object is the hand holding? Answer directly (e.g. 'Fork.', 'Cup.'). ONLY RETURN ONE OBJECT! \n"
)
WHAT_COLOR_IS_THE_OBJECT_PROMPT = (
    "What color is the object being held? Answer directly (e.g. 'Red.', 'Blue.').\n"
)
# Remove 'visibly' in the prompt. We'll use the motion blur check below.
RELEASE_PROMPT = (
    "This is a chunk from a video sequence. In the previous chunk, the hand was holding a(n) {target_object}. "
    "Answer directly: 'Yes.' if the hand lets go of the object in this chunk and the object is at least "
    "partially visible; answer 'No.' otherwise.\n"
)
GRASP_PROMPT = (
    "This is a chunk in a video sequence. The hand was not holding anything in the previous chunk. "
    "Answer directly: 'Yes.' if the hand grasps an object in this chunk and the object being "
    "grasped is visible; answer 'No.' otherwise.\n"
)
# Objects may get obscured due to motion blur. Check the last location where the object was not
# moving quickly. If it's there, we released. If not, we are still holding it.
CHECK_PREV_LOC_1_PROMPT = (
    "Is a(n) {target_object} visible? Answer 'Yes.' or 'No.' directly.\n"
)
CHECK_PREV_LOC_2_PROMPT = (
    "Is the {target_object} held by a hand? Answer 'Yes.' or 'No.' directly.\n"
)

CHECK_ACTIVE_INTERACTION_PROMPT = (
    "Does the hand actively interact with the {target_object} here? Answer 'Yes.' or 'No.' directly.\n"
    # "Does the hand make contact with the {target_object}? Answer 'Yes.' or 'No.' directly.\n"
)

##### --------------------------------- Nodes --------------------------------- #####


class ContactStateNode(ABC):
    """
    Stateless node object that *operates on* HandContactCtx and returns the next contact
    state.
    """

    def __init__(self, contact_ctx: HandContactCtx, vlm: VLMProtocol, **fmt):
        self.contact_ctx = contact_ctx
        self.vlm = vlm
        self.fmt = fmt  # formatting kwargs for prompts (e.g., held_object)

    def _query_vlm(self, frames: np.ndarray, prompt: str, **fmt) -> str:
        prompt = prompt.format(**fmt)
        ans = self.vlm.process_frames(frames, prompt)
        return ans

    def _repeat(self, pose_status: str) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        return (
            pose_status,
            deepcopy(self.contact_ctx.contacts),
            self.contact_ctx.held_object,
            {"method": "repeat", "outputs": "X -> X"},
        )

    def _recalibrate(
        self, pose_status: str, frames: np.ndarray
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        assert pose_status == HandStateStatus.OK
        ans = self._query_vlm(frames, BASIC_CONTACT_PROMPT, **self.fmt).lower()
        if "yes" in ans:
            held_object = (
                self._query_vlm(frames, WHICH_OBJECT_IN_CONTACT_PROMPT, **self.fmt)
                .strip()
                .lower()
            )
            held_object = ''.join([ch for ch in held_object if ch.isalpha() or ch.isspace()])
            if held_object:
                held_object_color = (
                    self._query_vlm(frames, WHAT_COLOR_IS_THE_OBJECT_PROMPT, **self.fmt)
                    .strip()
                    .lower()
                )
                held_object_color = ''.join([ch for ch in held_object_color if ch.isalpha()])
                held_object = f"{held_object_color} {held_object}".strip()
                return (
                    pose_status,
                    [True],
                    held_object,
                    {
                        "method": "recalibrate",
                        "outputs": f"X -> Hold {held_object} (Recalibrate). Ans: {ans}",
                    },
                )
        else:
            held_object = None
        return (
            pose_status,
            [False],
            None,
            {
                "method": "recalibrate",
                "outputs": f"X -> No items (Recalibrate). Ans: {ans}",
            },
        )

    def _state_dependent_prompt(
        self, pose_status: str, frames: np.ndarray
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        assert (
            pose_status == HandStateStatus.OK
            or pose_status == HandStateStatus.FAST_MOVEMENT
        )
        prev_contact = self.contact_ctx.contacts[-1]
        target_object = self.contact_ctx.held_object

        if prev_contact:
            ans = self._query_vlm(
                frames, RELEASE_PROMPT, **{**self.fmt, "target_object": target_object}
            ).lower()
            if "yes" in ans:
                # Released
                return (
                    pose_status,
                    [False],
                    None,
                    {
                        "method": "state_dependent_prompt",
                        "outputs": f"Hold {target_object} -> Release. Ans: {ans}",
                    },
                )
            return (
                pose_status,
                [True],
                target_object,
                {
                    "method": "state_dependent_prompt",
                    "outputs": f"Hold {target_object} -> Continue. Ans: {ans}",
                },
            )
        else:
            ans = self._query_vlm(frames, GRASP_PROMPT, **self.fmt).lower()
            if "yes" in ans:
                held_object = (
                    self._query_vlm(frames, WHICH_OBJECT_IN_CONTACT_PROMPT, **self.fmt)
                    .strip()
                    .lower()
                )
                held_object = ''.join([ch for ch in held_object if ch.isalpha() or ch.isspace()])
                if held_object and "no object" not in held_object:
                    # Color
                    held_object_color = (
                        self._query_vlm(frames, WHAT_COLOR_IS_THE_OBJECT_PROMPT, **self.fmt)
                        .strip()
                        .lower()
                    )
                    held_object_color = ''.join([ch for ch in held_object_color if ch.isalpha()])
                    held_object = f"{held_object_color} {held_object}".strip()

                    # Check active interaction to avoid false positives
                    ans2 = self._query_vlm(
                        frames,
                        CHECK_ACTIVE_INTERACTION_PROMPT,
                        **{**self.fmt, "target_object": held_object},
                    ).lower()

                    if "yes" in ans2:
                        return (
                            pose_status,
                            [True],
                            held_object,
                            {
                                "method": "state_dependent_prompt",
                                "outputs": f"No items -> Hold {held_object}. Ans: {ans} | {ans2}",
                            },
                        )
                else:
                    ans2 = "No object identified"
            else:
                ans2 = ""
            return (
                pose_status,
                [False],
                None,
                {
                    "method": "state_dependent_prompt",
                    "outputs": f"No items -> No items. Ans: {ans} | {ans2}",
                },
            )

    @staticmethod
    def _bbox_overlap(
        boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]
    ) -> float:
        # Compute the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union by taking the intersection area and
        # dividing it by the sum of prediction + ground-truth areas - the intersection area
        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0:
            return 0.0  # or raise ValueError("Degenerate boxes")
        return interArea / denom

    def _do_second_look_for_fast_movement(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        overlap_thresh: float = 0.1,
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        """
        Motion blur can obscure an object. Check the spot of the last OK.

        If the object is visible and NOT held by hand, we released.

        Arguments:
            pose_status: should be FAST_MOVEMENT
            frames_prev_loc: frames cropped to the last known OK box
        """
        assert pose_status == HandStateStatus.FAST_MOVEMENT or pose_status == HandStateStatus.OK
        prev_contact = self.contact_ctx.contacts[-1]
        assert prev_contact == True

        last_box = self.contact_ctx.last_ok_box
        last_boxes = [last_box] * len(frames)
        cropped_frames_prev_loc = _get_cropped(frames, last_boxes)

        target_object = self.contact_ctx.held_object

        ans1 = self._query_vlm(
            cropped_frames_prev_loc,
            CHECK_PREV_LOC_1_PROMPT,
            **{**self.fmt, "target_object": target_object},
        ).lower()
        if "yes" in ans1:
            # Sufficient overlap between the current box and last known OK box?
            max_overlap = max([self._bbox_overlap(b1, last_box) for b1 in bboxes])
            if max_overlap < overlap_thresh:
                # The hand moved away, the object is visible in prev location, we must have released
                return (
                    pose_status,
                    [False],
                    None,
                    {
                        "method": "second_look_fast_movement",
                        "outputs": f"Hold {target_object} -> Release (2nd look). Ans: {ans1}. Max overlap: {max_overlap:.2f}",
                    },
                )

            # There is sufficient overlap, check if the object is actually held
            ans2 = self._query_vlm(
                cropped_frames_prev_loc,
                CHECK_PREV_LOC_2_PROMPT,
                **{**self.fmt, "target_object": target_object},
            ).lower()
            if "no" in ans2:
                # We must have released
                return (
                    pose_status,
                    [False],
                    None,
                    {
                        "method": "second_look_fast_movement",
                        "outputs": f"Hold {target_object} -> Release (2nd look). Ans: {ans1} | {ans2}",
                    },
                )
        else:
            ans2 = ""
        return (
            pose_status,
            [True],
            target_object,
            {
                "method": "second_look_fast_movement",
                "outputs": f"Hold {target_object} -> Continue (2nd look). Ans: {ans1} | {ans2}",
            },
        )

    def _state_dependent_prompt_with_motion_blur_guard(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        """
        Arguments:
            pose_status: should be OK or FAST_MOVEMENT
            frames: cropped frames of current boxes
            frames_prev_loc: cropped frames of last known OK box (for motion blur guard)
        """
        cropped_frames = _get_cropped(frames, bboxes)
        pose_status, contacts, held_object, info = self._state_dependent_prompt(
            pose_status, cropped_frames
        )

        # Motion blur occurs when we were previously in contact and the current
        # contact says we released b/c the item is obscured by motion.
        # We only proceed to motion blur guard if:
        # 1) We were previously in contact
        # 2) We are not currently in contact as determined by the VLM
        # 3) We have a last known OK box to check against
        # 4) The current pose status is FAST_MOVEMENT (if OK, we trust the VLM)
        last_box = self.contact_ctx.last_ok_box
        prev_contact = self.contact_ctx.contacts[-1]
        motion_blur_guard_needed = (
            (prev_contact == True)
            and (contacts[-1] == False)
            and (last_box is not None)
            and (pose_status == HandStateStatus.FAST_MOVEMENT)
        )

        if motion_blur_guard_needed:
            # Guard against motion blur
            return self._do_second_look_for_fast_movement(pose_status, frames, bboxes)
        else:
            return pose_status, contacts, held_object, info

    @abstractmethod
    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        """
        Execute this node:
        Returns: (cur_state_status, cur_state_contact, cur_state_held_object, info)
        """
        ...


ContactStateNodeType = Type[ContactStateNode]
CONTACT_STATE_REGISTRY: Dict[str, ContactStateNodeType] = (
    {}
)  # filled by @register_contact_node below


def register_contact_node(state_name: str):
    def _wrap(cls: ContactStateNodeType) -> ContactStateNodeType:
        CONTACT_STATE_REGISTRY[state_name] = cls
        cls.state_name = state_name  # for convenience
        return cls

    return _wrap


@register_contact_node(HandStateStatus.ABSTAIN)
class AbstainNode(ContactStateNode):
    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        if (
            pose_status == HandStateStatus.ABSTAIN
            or pose_status == HandStateStatus.FAST_MOVEMENT
        ):
            return self._repeat(pose_status)

        cropped_frames = _get_cropped(frames, bboxes)
        return self._recalibrate(pose_status, cropped_frames)


@register_contact_node(HandStateStatus.FAST_MOVEMENT)
class FastMovementNode(ContactStateNode):
    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        if pose_status == HandStateStatus.ABSTAIN:
            return self._repeat(pose_status)

        return self._state_dependent_prompt_with_motion_blur_guard(
            pose_status, frames, bboxes
        )


@register_contact_node(HandStateStatus.OK)
class OKNode(ContactStateNode):
    def run(
        self,
        pose_status: str,
        frames: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:
        if pose_status == HandStateStatus.ABSTAIN:
            return self._repeat(pose_status)

        return self._state_dependent_prompt_with_motion_blur_guard(
            pose_status, frames, bboxes
        )


####################################### ORCHESTRATION SECTION #######################################


class HandStateMachine:
    def __init__(self, *, handedness: str):
        self.contact_ctx = HandContactCtx(handedness=handedness)
        self.idle_ctx = HandIdleCtx(handedness=handedness)

    def _make_contact_node(
        self, vlm: VLMProtocol, *, target_objects: str
    ) -> ContactStateNode:
        node_cls = CONTACT_STATE_REGISTRY[self.contact_ctx.status]
        # Pass current known object for formatting. Caller can add more via **extra_fmt.
        return node_cls(
            self.contact_ctx,
            vlm,
            target_objects=target_objects,
            handedness=self.contact_ctx.handedness,
        )

    def _make_idle_node(self, vlm: VLMProtocol) -> IdleStateNode:
        return IdleStateNode(self.idle_ctx, vlm)

    def step(
        self, chunk: VideoChunk, vlm: VLMProtocol, *, target_objects: str = ""
    ) -> Tuple[str, List[bool], Optional[str], Dict[str, Any]]:

        # CONTACT
        contact_node = self._make_contact_node(vlm, target_objects=target_objects)
        status, contacts, obj, info_contact = contact_node.run(
            chunk.pose_status, chunk.frames, chunk.bboxes
        )
        self.contact_ctx.status = status  # Update context
        self.contact_ctx.contacts = contacts
        self.contact_ctx.held_object = obj
        if status == HandStateStatus.OK:
            self.contact_ctx.last_ok_box = chunk.bboxes[-1]  # last frame's box
        if contacts[-1] == True:
            self.contact_ctx.last_contact_box = chunk.bboxes[-1]  # last frame's box

        # IDLE
        idle_node = self._make_idle_node(vlm)
        idles, info_idle = idle_node.run(chunk.pose_status, chunk.frames, chunk.bboxes)
        self.idle_ctx.idles = idles

        return status, contacts, obj, idles, info_contact, info_idle

    @property
    def contact_context(self) -> HandContactCtx:
        return self.contact_ctx

    @property
    def idle_context(self) -> HandIdleCtx:
        return self.idle_ctx


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

    machine = HandStateMachine(handedness=handedness)
    target_objects = _get_target_objects(video_path)

    all_contacts, all_bboxes, all_status, all_objs, all_times = [], [], [], [], []
    all_idles = []
    all_kps_wrist, all_kps_elbow, all_kps_hand = [], [], []

    hand_locator = HandLocator(pose_stream, vlm)
    hand_cropper = HandCropper(
        kp_conf_thresh=0.90, fast_movement_thresh=12, interpolation="mixed"
    )
    hand_locator.clear()
    hand_cropper.clear()

    for frames, start_t, end_t in load_long_video_decord(
        video_path,
        max_frames_num=max_frames_num,
        sampling_strategy=sampling_strategy,
        overlap_frames_num=overlap_frames_num,
        sampling_fps=sampling_fps,
        force_sample=False,
        ret_idx=False,
    ):
        # Locate hand
        kps = hand_locator.process_frames(frames, handedness=handedness)
        kps_wrist, kps_elbow, kps_hand = kps["wrist"], kps["elbow"], kps["hand"]

        # Crop
        pose_status, boxes = hand_cropper.process_frames(frames, hand_kps=kps_hand)

        # State machine step
        chunk = VideoChunk(
            pose_status=pose_status,
            frames=frames,
            bboxes=boxes,
            start_t=start_t,
            end_t=end_t,
        )

        # status, contacts, obj, idles, info_contact, info_idle
        status, contacts, obj, idles, info_contact, info_idle = machine.step(
            chunk, vlm, target_objects=target_objects
        )

        # logger.info(
        #     f"{start_t:.2f}-{end_t:.2f}s | {info_contact['method']} | {info_contact['outputs']} | pose conf: {min([kps_hand[i][2] for i in range(len(kps_hand))]):.3f} | {info_idle['method']} | {info_idle['outputs']}"
        # )
        print(
            f"{start_t:.2f}-{end_t:.2f}s | {info_contact['method']} | {info_contact['outputs']} | pose conf: {min([kps_hand[i][2] for i in range(len(kps_hand))]):.3f} | {info_idle['method']} | {info_idle['outputs']}"
        )

        num_idles = len(idles)
        num_contacts = len(contacts)
        num_frames = len(boxes)

        all_times.extend(np.linspace(start_t, end_t, num_frames).tolist())
        all_bboxes.extend(boxes)
        all_status.extend([status] * num_frames)
        all_objs.extend([obj] * num_frames)
        all_kps_wrist.extend([kps_wrist[i] for i in range(num_frames)])
        all_kps_elbow.extend([kps_elbow[i] for i in range(num_frames)])
        all_kps_hand.extend([kps_hand[i] for i in range(num_frames)])
        contacts_idx = np.linspace(
            0, num_contacts - 1, num_frames, dtype=np.int32
        ).tolist()
        all_contacts.extend([contacts[ci] for ci in contacts_idx])
        idles_idx = np.linspace(0, num_idles - 1, num_frames, dtype=np.int32).tolist()
        all_idles.extend([idles[ii] for ii in idles_idx])

    infos = {
        "idles": all_idles,
        "contacts": all_contacts,
        "bboxes": all_bboxes,
        "status": all_status,
        "objs": all_objs,
        "times": all_times,
        "kps_wrist": all_kps_wrist,
        "kps_elbow": all_kps_elbow,
        "kps_hand": all_kps_hand,
    }

    # Integrate idles and contacts to primitives
    assert len(all_idles) == len(all_contacts)
    N = len(all_idles)

    # -------- Postprocessing -------

    # 1) Contact takes priority
    prims = ["transport" if contact else "UNK" for contact in all_contacts]

    # 2) Verify the idle: we need at least `num_frames_for_idle` consecutive frames to declare an idle
    num_prev_idle = [0] * N
    num_prev_idle[0] = 1 if all_idles[0] else 0
    for i in range(1, N):
        if all_idles[i]:
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
        "idle" if (not contact and verified_idle) else prim
        for prim, verified_idle, contact in zip(prims, verified_idles, all_contacts)
    ]

    # 3) Determine between reach, reposition, and reposition-reach
    contact_in_future = [False] * N
    contact_state = False
    for i in range(N - 1, -1, -1):
        if contact_state:
            if prims[i] == "idle":
                contact_state = False
        else:
            if prims[i] == "transport":
                contact_state = True
        contact_in_future[i] = contact_state
    contact_in_past = [False] * N
    contact_state = False
    for i in range(N):
        if contact_state:
            if prims[i] == "idle":
                contact_state = False
        else:
            if prims[i] == "transport":
                contact_state = True
        contact_in_past[i] = contact_state

    # Go through prims
    # - If no future contact -> reposition
    # - If no prior contact and future contact -> reach
    # - If prior contact and future contact -> reposition-reach
    for i in range(N):
        if prims[i] == "UNK":
            if not contact_in_future[i]:
                prims[i] = "reposition"
            elif not contact_in_past[i]:
                prims[i] = "reach"
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
            prims[i:j] = ["reach"] * L  # too short, just reach
        else:
            first_half = (L - 1) // 2 + 1
            second_half = L - first_half
            prims[i : i + first_half] = ["reposition"] * first_half
            prims[i + first_half : j] = ["reach"] * second_half

        i = j

    # 4) Smooth out quick idles -> contacts and contacts -> idles 
    # with a reach/reposition
    side = max_frames_num // 2
    for i in range(1, N):
        if prims[i-1] == "idle" and prims[i] == "transport":
            for j in range(max(0, i-side), min(N, i+side)):
                prims[j] = "reach"
        
        if prims[i-1] == "transport" and prims[i] == "idle":
            for j in range(max(0, i-side), min(N, i+side)):
                prims[j] = "reposition"


    return prims, all_times, infos
