import decord
import json
import os

from data.utils_strokerehab import (
    HUMAN_INPUT_JSON_PATH,
    HUMAN_INPUT_JSON_PATH_BACKUP,
    strokerehab_load_dataset,
    LabelUtils,
    LABEL_DIR,
    VIDEO_DIR
)



class HumanInputDataManager:
    """
    Human input JSON format:

    Dictionary with video path as key and a dictionary as values.

    The inner dictionary has frame indices as key and a dictionary
    containing the frame annotations.
    """
    def __init__(self, annotation_frequency_s: float = 10000, sampling_fps: int = 8):
        """
        `annotation_frequency_s` is the frequency of annotation in seconds.
        """

        # hardcode the dataset as the one in strokerehab dataset
        self.dataset = strokerehab_load_dataset(filter_for_testset=True)
        if 'test' in self.dataset:
            self.dataset = self.dataset['test']
        # has `path_v` for video path and `duration_s` for video duration, `fps` for video fps

        try:
            with open(HUMAN_INPUT_JSON_PATH, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

        # create a backup of the human input
        os.makedirs(os.path.dirname(HUMAN_INPUT_JSON_PATH_BACKUP), exist_ok=True)
        with open(HUMAN_INPUT_JSON_PATH_BACKUP, 'w') as f:
            json.dump(self.data, f)

        self.annotation_frequency_s = annotation_frequency_s
        self.sampling_fps = sampling_fps

        # list tuple of (video path, frame index)
        self.frames_needed = self._calculate_frames_needed()
        self.frames_annotated = set()

        # Track to do less I/O operations
        self.cur_idx = 0
        self.cur_video_path = None
        self.video_reader = None

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
            done_frames = set()
            if path_v in self.data:
                # Extract frame indices that have been annotated
                done_frames = set(int(frame_idx) for frame_idx in self.data[path_v].keys())
            
            # Calculate sampling rate based on original video fps and desired sampling fps
            sampling_rate = int(round(video_fps / self.sampling_fps))
            
            # Calculate total number of frames in the video
            total_frames = int(round(duration_s * video_fps))
            
            # Calculate frame indices to sample
            sampled_frames = list(range(0, total_frames, sampling_rate))
            
            # Ensure we have at least one frame per annotation_frequency_s seconds
            max_frame_gap_s = self.annotation_frequency_s
            max_frame_gap = int(round(max_frame_gap_s * video_fps))

            frames_needing_ann = []
            for sampled_frame in sampled_frames:
                if not frames_needing_ann:
                    frames_needing_ann.append(sampled_frame)
                else:
                    # Check if the current sampled frame is too far from the last annotated frame
                    if sampled_frame - frames_needing_ann[-1] > max_frame_gap \
                        and sampled_frame not in done_frames:
                        frames_needing_ann.append(sampled_frame)
            frames_needed.extend([(path_v, handedness, frame_idx) for frame_idx in frames_needing_ann])

        return frames_needed
    
    def current(self):
        path_v, handedness, frame_idx = self.frames_needed[self.cur_idx]
        if path_v != self.cur_video_path:
            self.cur_video_path = path_v
            self.video_reader = decord.VideoReader(path_v)
        frame = self.video_reader[frame_idx].asnumpy()
        return {
            'path_v': path_v,
            'frame_idx': frame_idx,
            'frame': frame,
            'handedness': handedness,
            'num_frames_needed': len(self.frames_needed),
            'num_frames_done': len(self.frames_annotated),
        }
    
    def _done_signal(self):
        return {
            'num_frames_needed': len(self.frames_needed),
            'num_frames_done': len(self.frames_annotated),
        }

    def next(self):
        if len(self.frames_needed) == len(self.frames_annotated):
            return self._done_signal()
        self.cur_idx = (self.cur_idx + 1) % len(self.frames_needed)
        while self.cur_idx in self.frames_annotated:
            self.cur_idx = (self.cur_idx + 1) % len(self.frames_needed)
        return self.current()

    def prev(self):   
        if len(self.frames_needed) == len(self.frames_annotated):
            return self._done_signal()
        self.cur_idx = (self.cur_idx - 1) % len(self.frames_needed)
        while self.cur_idx in self.frames_annotated:
            self.cur_idx = (self.cur_idx - 1) % len(self.frames_needed)
        return self.current()
    
    def annotate_cur(self, path_v, frame_idx, result):
        if path_v not in self.data:
            self.data[path_v] = {}
        self.data[path_v][frame_idx] = result
        self.frames_annotated.add(self.cur_idx)

    def save(self):
        os.makedirs(os.path.dirname(HUMAN_INPUT_JSON_PATH), exist_ok=True)
        with open(HUMAN_INPUT_JSON_PATH, 'w') as f:
            json.dump(self.data, f)
