import json
import numpy as np
import torch
from typing import Union, Dict, List, Optional, Any

import numpy as np
import cv2
import os

import logging

from vidplot.encode.segmentations import encode_segmentation_masks

from vidplot import AnnotationOrchestrator
from vidplot.streamers import VideoStreamer, TabularStreamer
from vidplot.renderers import RGBRenderer, SegmentationRenderer

from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_camera_predictor import SAM2CameraPredictor

from data.utils_strokerehab import (
    LabelUtils,
    DataPaths,
    strokerehab_load_dataset,
)

from pathlib import Path


# logger at info level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('sam2_tracking.log', mode='w')
    ]
)
# Create a logger for this module
logger = logging.getLogger(__name__)


# Setup torch precision and performance
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class NothingToTrackError(Exception):
    """
    Custom exception raised when there are no hand bounding boxes to track.
    """
    def __init__(self, message: str = "No hand bounding boxes found to track."):
        super().__init__(message)


class SAM2PromptHinter:
    """
    Loads hand bounding box hints from JSON and provides query capabilities.
    """
    def __init__(self, json_source: Union[str, Dict[str, Any]]):
        if isinstance(json_source, str):
            with open(json_source, 'r') as f:
                self.hints = json.load(f)
        else:
            self.hints = json_source

    def get_hand_bboxes(
        self,
        video_path: str,
        frame_number: int
    ) -> Dict[str, Optional[List[int]]]:
        """
        Retrieve left and right hand bboxes for a specific frame.
        """
        video_hints = self.hints.get(video_path, {})
        frame_hints = video_hints.get(str(frame_number), {})
        return {
            'left_hand': frame_hints.get('left_hand'),
            'right_hand': frame_hints.get('right_hand')
        }

    def get_all_hand_bboxes(
        self,
        video_path: str
    ) -> Dict[int, Dict[str, Optional[List[int]]]]:
        """
        Retrieve hand bboxes for all frames in a video that have hints.

        Returns:
            A dict mapping frame_number (int) to a dict with keys 'left_hand' and 'right_hand'.
        """
        video_hints = self.hints.get(video_path, {})
        all_bboxes: Dict[int, Dict[str, Optional[List[int]]]] = {}
        for frame_str, frame_hints in video_hints.items():
            try:
                frame_idx = int(frame_str)
            except ValueError:
                continue
            all_bboxes[frame_idx] = {
                'left_hand': frame_hints.get('left_hand'),
                'right_hand': frame_hints.get('right_hand')
            }
        return dict(sorted(all_bboxes.items()))
    

def track_with_hints(
    video_path: str,
    hinter: SAM2PromptHinter,
    predictor: SAM2CameraPredictor,
    segmentations_out_path: str,
    save_first_frame_only: bool = False,
):
    """
    Track hands in a video using SAM2 with prompts from a JSON file containing
    pre-collected hand bounding boxes.

    This version uses cv2 for video I/O and the encoding functions from
    `segmentations.py`, saving metadata only for the first frame with
    segmentations, and providing an option to process and save only that first frame.

    Args:
        video_path (str): Path to the input video file.
        hinter (SAM2PromptHinter): Instance of SAM2PromptHinter to provide hand
            bounding boxes.
        predictor (SAM2CameraPredictor): Instance of SAM2CameraPredictor to perform
            tracking.
        segmentations_out_path (str): Path to save the output segmentations.
        save_first_frame_only (bool): If True, tracking and saving will stop
            after the first frame with detected objects. This is useful for
            quick previews.

    Raises:
        IOError: If the video file cannot be opened.
        NothingToTrackError: If no hand bounding boxes are found in the video.
    """
    # Video I/O setup using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup storage
    video_encoded_timestamps = []
    video_encoded_masks = []

    # Build all prompts by seeking to specific frames
    boxes = hinter.get_all_hand_bboxes(video_path)
    first_frame_idx, first_obj_ids, first_mask_logits = None, None, None
    for frame_idx, frame_boxes in sorted(boxes.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {frame_idx} during prompt building.")
            continue
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictor.load_frame(frame_idx, frame_rgb)
        
        # Add prompt(s)
        for hand, obj_id in [('left_hand', 1), ('right_hand', 2)]:
            bbox = frame_boxes.get(hand)
            if bbox:
                bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    bbox=bbox_arr,
                )
        if first_frame_idx is None:
            first_frame_idx = frame_idx
            first_obj_ids = out_obj_ids
            first_mask_logits = out_mask_logits
    
    if first_frame_idx is None:
        cap.release()
        raise NothingToTrackError("No hand bounding boxes found in the video.")

    # Reset video capture to the beginning for sequential tracking
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Track hands throughout video
    for idx in range(num_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {idx}. Stopping.")
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Get masks for the current frame
        if idx == first_frame_idx:
            out_obj_ids, out_mask_logits = first_obj_ids, first_mask_logits
        elif idx < first_frame_idx:
            out_obj_ids, out_mask_logits = [], []
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame_rgb)
        
        out_masks = [
            (mask_logit > 0).squeeze().cpu().numpy().astype(np.uint8)
            for mask_logit in out_mask_logits
        ]  # List of HxW masks

        # Encode masks. Metadata is saved only for the first frame with segmentations.
        is_first_frame_with_segs = (idx == first_frame_idx)
        encoded_masks = encode_segmentation_masks(
            seg_ids=out_obj_ids,
            seg_masks=out_masks,
            save_metadata=is_first_frame_with_segs
        )

        video_encoded_timestamps.append(round(float(idx) / fps if fps > 0 else 0.0, 3))
        video_encoded_masks.append(encoded_masks)

        # Logging
        hints = hinter.get_hand_bboxes(video_path, idx)
        if idx % 200 == 0 or hints.get('left_hand') or hints.get('right_hand'):
            logger.info(f"Processing frame {idx}/{num_frames} with hints: {hints}")
        
        # If only saving the first frame, break after processing it.
        if save_first_frame_only and idx == first_frame_idx:
            logger.info(f"Stopping after processing the first hinted frame ({idx}) as requested.")
            break

    # Release the video capture object
    cap.release()

    # Finish saving the payload to a JSON file
    payload = {
        'timestamps': video_encoded_timestamps,
        'masks': video_encoded_masks,
    }

    with open(segmentations_out_path, 'w') as f:
        json.dump(payload, f)
    
    logger.info(f"Saved segmentation data to {segmentations_out_path}")


def annotate_video_with_segmentations(
    video_path: str,
    label_path: str,
    segmentations_out_path: str,
    out_path: str,
):
    """
    Annotate a video with SAM 2 segmentations and save it to out_path.
    """

    video_streamer = VideoStreamer("video_stream", video_path, backend="opencv", sample_rate=30)
    segmentations_streamer = TabularStreamer("seg_stream", segmentations_out_path, "masks", "timestamps", sample_rate=30)

    renderer = RGBRenderer("rgb_renderer", video_streamer, grid_row=(1,1), grid_column=(1,1))
    segmentation_renderer = SegmentationRenderer(
        "seg_renderer", segmentations_streamer, id_to_color={1: (255, 0, 0)}, alpha=0.3,  # color in red
        grid_row=(1,1), grid_column=(1,1), z_index=1,
    )

    width, height = video_streamer.size
    orchestrator = AnnotationOrchestrator(
        grid_template_rows=[height],
        grid_template_columns=[width]
    )
    orchestrator.set_annotators(
        [video_streamer, segmentations_streamer],
        [renderer, segmentation_renderer],
        routes=[("video_stream", "rgb_renderer"), ("seg_stream", "seg_renderer")],
    )
    orchestrator.write(out_path, fps=30)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Track hands in videos using SAM2 with prompts from a JSON file.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Just do the first frame."
    )
    args = parser.parse_args()
    if args.debug:
        ext = "png"
        save_first_frame_only = True
    else:
        ext = "mp4"
        save_first_frame_only = False

    dataset = strokerehab_load_dataset(
        video_regex=r"C00015_\w+1_1"
    )
    label_paths = dataset['test']['path_l']
    video_paths = dataset['test']['path_v']

    # Initialize hinter and predictor
    SAM2_CKPT = "./SAM2/checkpoints/sam2.1_hiera_base_plus.pt"
    SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CKPT)

    for video_path, label_path in zip(video_paths, label_paths):

        video_path_no_ext = video_path.split('.')[0]
        video_path = os.path.join(DataPaths.RAW_VIDEO_DIR, video_path)
        label_path = os.path.join(DataPaths.RAW_LABEL_DIR, label_path)

        # JSON file for segmentations
        segmentations_out_path = Path(DataPaths.SAM2_ANNOTATED_VIDEOS_PATH) / f"{video_path_no_ext}_tracked.json"
        if not segmentations_out_path.parent.exists():
            segmentations_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Video of segmentations
        out_path = Path(DataPaths.SAM2_ANNOTATED_VIDEOS_PATH) / f"{video_path_no_ext}_tracked.{ext}"
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            hinter = SAM2PromptHinter(DataPaths.HUMAN_INPUT_JSON_PATH)
            track_with_hints(
                video_path=video_path,
                hinter=hinter,
                predictor=predictor,
                segmentations_out_path=str(segmentations_out_path),
                save_first_frame_only=save_first_frame_only
            )
            annotate_video_with_segmentations(
                video_path=video_path,
                label_path=str(label_path),
                segmentations_out_path=str(segmentations_out_path),
                out_path=str(out_path)
            )
            logging.info(f"Tracking completed for {video_path}. Output saved to {out_path}")
        except NothingToTrackError as e:
            logging.error(f"Nothing to track in {video_path}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while processing {video_path}: {e}")
