"""
Contains utility classes/functions for specifying data paths, retrieving videos
and labels metadata, and extracting action information.
"""

import multiprocessing as mp
from collections import defaultdict
import os
from typing import Callable, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.utils_strokerehab import DataPaths
from decord import VideoReader, cpu



def return_np_save_path(label_file_path, save_dir):
    """Determines path to save np file. Mimics raw labels folder structure."""
    label_file_path = os.path.relpath(label_file_path, DataPaths.RAW_LABEL_DIR)
    folder_name, file_name = os.path.split(label_file_path)
    os.makedirs(os.path.join(save_dir, folder_name), exist_ok=True)
    np_name = file_name.replace('.csv', '.npz')
    return os.path.join(save_dir, folder_name, np_name)


def split_path(path):
    parts = []
    while True:
        path, tail = os.path.split(path)
        if tail:
            parts.insert(0, tail)
        else:
            if path:
                parts.insert(0, path)
            break
    return parts


def ensure_dir_exists(path):
    """Ensures directory exists."""
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)


def return_abs_paths(directory):
    """Returns the absolute paths of all files in a directory."""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def extract_folder_file_from_path(label_file_path):
    """Converts label file path to folder and file name."""
    arr = split_path(label_file_path)
    folder_name = arr[-2]
    file_name = arr[-1]
    return folder_name, file_name


def preprocess_frames(frames):
    """Converts a list of HWC (BGR) frames to a normalized BCHW tensor (RGB)."""
    frames_np = np.stack(frames, axis=0)
    frames_torch = torch.FloatTensor(frames_np).permute(0, 3, 1, 2)  # BHWC -> BCHW
    return frames_torch.flip(1) / 255.  # BGR to RGB


def identity_frames_np(frames):
    """Pass frames through: BHWC (BGR)."""
    return np.stack(frames, axis=0)


def yield_frames(video_path, chunk_size,
                 starting_frame=-1, ending_frame=-1,
                 preprocess_frames_fn = identity_frames_np):
    """
    Yield video frames in chunks as a normalized BCHW tensor (RGB).
    If `begin_frame` and `end_frame` are not specified, the entire video is read.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    starting_frame = max(0, starting_frame)
    ending_frame = ending_frame if ending_frame > 0 else float('inf')
    num_frames_to_read = ending_frame - starting_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

    while True and num_frames_to_read > 0:
        ret, frame = cap.read()
        if not ret:
            break

        num_frames_to_read -= 1
        frames.append(frame)

        if len(frames) == chunk_size:
            yield preprocess_frames_fn(frames)
            frames.clear()

    if len(frames) > 0:
        yield preprocess_frames_fn(frames)

    cap.release()


def collate_metadata(extract_metadata_fn: Callable, paths: List[str]) -> Dict[str, List]:
    """Uses multiprocessing to gather metadata from a list of paths (e.g., videos, csv's).

    Args:
    - extract_metadata_fn: function that returns a dictionary of info given a path. Include
        the path in the dictionary.
    - paths: list of paths to extract metadata from

    Returns:
    - collated_info: dictionary of lists of metadata (unordered)
    """
    with mp.Pool(mp.cpu_count()) as p:
        infos = []
        # append unordered results for speed
        for info in tqdm(p.imap_unordered(extract_metadata_fn, paths), total=len(paths)):
            if info is not None:
                infos.append(info)
        p.close()
        p.join()

    collated_info = defaultdict(list)
    for info in infos:
        for key, value in info.items():
            collated_info[key].append(value)

    return dict(collated_info)


def write_metadata(data_dir, out_path, metadata_fn):
    paths = return_abs_paths(data_dir)
    df = pd.DataFrame(collate_metadata(metadata_fn, paths))
    ensure_dir_exists(out_path)
    df.to_csv(out_path, index=False)


def get_first_frame(path):
    """Returns the first frame of a video."""
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    return frame


def show_first_frame(path):
    """Displays the first frame of a video using matplotlib."""
    import matplotlib.pyplot as plt
    frame = get_first_frame(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    plt.imshow(frame)
    plt.axis('off')
    return frame
