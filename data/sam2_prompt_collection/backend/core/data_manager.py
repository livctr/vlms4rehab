import decord
import logging
import json
import os
from typing import Optional

from data.utils_strokerehab import (
    HUMAN_INPUT_JSON_PATH,
    strokerehab_load_dataset,
    LabelUtils,
    LABEL_DIR,
    VIDEO_DIR
)



class DataManager:
    """
    Data manager for obtaining bounding boxes prompts for SAM 2.

    The inner dictionary has frame indices as key and a dictionary
    containing the frame annotations.
    """
    def __init__(self, storage_path: str = HUMAN_INPUT_JSON_PATH,
                 annotation_frequency_s: float = 5,
                 sampling_fps: int = 8,
                 dataset_kwargs: Optional[dict] = None):
        """
        `annotation_frequency_s` is the frequency of annotation in seconds.
        """

        # hardcode the dataset as the one in strokerehab dataset
        # has `path_v` for video path and `duration_s` for video duration, `fps` for video fps
        dataset_kwargs = dataset_kwargs or {"filter_for_testset": True}
        self.dataset = strokerehab_load_dataset(**dataset_kwargs)
        if 'test' in self.dataset:
            self.dataset = self.dataset['test']

        self.storage_path = storage_path
        try:
            with open(storage_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

        backup_fname = os.path.basename(storage_path).split('.')[0] + '_backup.json'
        backup_path = os.path.join(
            os.path.dirname(storage_path),
            backup_fname
        )
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(backup_path, 'w') as f:
            json.dump(self.data, f, indent=4)

        self.annotation_frequency_s = annotation_frequency_s
        self.sampling_fps = sampling_fps

        # list tuple of (video path, frame index)
        self.frames_needed = self._calculate_frames_needed()
        self.frames_annotated = set()

        # Track to do less I/O operations
        self.cur_idx = 0
        self.cur_video_path = None
        self.cur_video_frame_idx = None
        self.video_reader = None
        self.cur_frame = None
        self.cur_handedness = None
    
    @property
    def total_frames_needed(self):
        return len(self.frames_needed)

    def done(self):
        return len(self.frames_needed) == len(self.frames_annotated)

    def _calculate_frames_needed(self):
        """
        Calculate which frames in each video need human annotation based on:
        1. Sampling rate derived from video fps and desired sampling fps
        2. Annotation frequency requirement
        
        Returns a dictionary mapping video paths to lists of frame indices that need annotation.
        """
        frames_needed = []
        handedness_dict = {}
        
        for data_item in self.dataset:
            path_v = os.path.join(VIDEO_DIR, data_item['path_v'])
            path_l = os.path.join(LABEL_DIR, data_item['path_l'])
            if path_l not in handedness_dict:
                handedness_dict[path_l] = LabelUtils.get_handedness(path_l)
            handedness = handedness_dict[path_l]
            video_fps = data_item['fps']
            duration_s = data_item['duration_s']

            # Get frame indices that already have annotations
            done_frame_idxs = set()
            if path_v in self.data:
                # Extract frame indices that have been annotated
                done_frame_idxs = set(int(frame_idx) for frame_idx in self.data[path_v].keys())

            # Calculate sampling rate based on original video fps and desired sampling fps
            sampling_rate = int(round(video_fps / self.sampling_fps))
            
            # Calculate total number of frames in the video
            total_num_frames = int(round(duration_s * video_fps))
            
            # Calculate frame indices to sample
            sampled_frame_idxs = list(range(0, total_num_frames, sampling_rate))
            
            # Ensure we have at least one frame per annotation_frequency_s seconds
            max_frame_gap_s = self.annotation_frequency_s
            max_frame_gap = int(round(max_frame_gap_s * video_fps))

            frames_needing_ann_pre = []
            for sampled_frame_idx in sampled_frame_idxs:
                if not frames_needing_ann_pre:
                    frames_needing_ann_pre.append(sampled_frame_idx)
                else:
                    # Check if the gap between the current and last annotated frame is too large
                    if sampled_frame_idx - frames_needing_ann_pre[-1] > max_frame_gap:
                        frames_needing_ann_pre.append(sampled_frame_idx)
            
            # Filter out the done set
            frames_needing_ann = []
            for sampled_frame_idx in frames_needing_ann_pre:
                if sampled_frame_idx not in done_frame_idxs:
                    frames_needing_ann.append(sampled_frame_idx)
            frames_needed.extend([(path_v, handedness, frame_idx) for frame_idx in frames_needing_ann])

        return frames_needed

    def _set_video_and_frame(self):
        """
        Set the current video and frame index.
        """
        path_v = self.frames_needed[self.cur_idx][0]
        handedness = self.frames_needed[self.cur_idx][1]
        frame_idx = self.frames_needed[self.cur_idx][2]
        if self.video_reader is None or path_v != self.cur_video_path:
            self.video_reader = decord.VideoReader(path_v)
        self.cur_video_path = path_v
        self.cur_handedness = handedness
        self.cur_video_frame_idx = frame_idx
        try:
            self.cur_frame = self.video_reader[frame_idx].asnumpy()
        except IndexError:
            self.cur_frame = None

    def current(self):
        if self.done():
            raise Exception("Calling current() when all frames are annotated.")

        self._set_video_and_frame()
        return {
            'path_v': self.cur_video_path,
            'frame_idx': self.cur_video_frame_idx,
            'frame': self.cur_frame,
            'handedness': self.cur_handedness,
            'num_frames_needed': len(self.frames_needed),
            'num_frames_done': len(self.frames_annotated),
        }
        
    def next(self):
        if self.done():
            return
        self.cur_idx = (self.cur_idx + 1) % len(self.frames_needed)
        while self.cur_idx in self.frames_annotated:
            self.cur_idx = (self.cur_idx + 1) % len(self.frames_needed)

    def prev(self):
        if self.done():
            return
        self.cur_idx = (self.cur_idx - 1) % len(self.frames_needed)
        while self.cur_idx in self.frames_annotated:
            self.cur_idx = (self.cur_idx - 1) % len(self.frames_needed)

    def annotate_cur(self, result, path_v=None, frame_idx=None):
        if path_v is None:
            path_v = self.cur_video_path
        if frame_idx is None:
            frame_idx = self.cur_video_frame_idx

        if self.cur_video_path not in self.data:
            self.data[self.cur_video_path] = {}
        self.data[self.cur_video_path][self.cur_video_frame_idx] = result
        self.frames_annotated.add(self.cur_idx)

    def save(self):
        logging.info(f"Saving human input data to {self.storage_path}")
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f, indent=4)
