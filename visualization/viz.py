"""Utils for visualizing video data. Fast(-er) with multiprocessing."""

from functools import partial
import logging
from multiprocessing import Process, Queue, Event
import os
from typing import Tuple, Callable, Optional

import cv2
import numpy as np
from tqdm import tqdm

from data.utils import ensure_dir_exists
from data.time_alignment.aligner import Aligner, TimeAligner
from data.time_alignment.data_iterator import DecordVideoIterator, PandasIterator

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

QUEUE_MAX_SIZE = 100


########################## Frame-Level Functions #####################********

def thumbnail(
        frame: np.ndarray, target_size: int = 512
) -> np.ndarray:
    scale_factor = min(target_size / frame.shape[0], target_size / frame.shape[1])
    new_height = int(frame.shape[1] * scale_factor)
    new_width = int(frame.shape[0] * scale_factor)
    return cv2.resize(frame, (new_height, new_width),
                        interpolation=cv2.INTER_LANCZOS4)


def pad_to_square(
        frame: np.ndarray, target_size: int = 512, fill_color: int = 255
) -> np.ndarray:
    # Get the original dimensions
    axis_dims = frame.shape
    assert len(axis_dims) == 3 and axis_dims[-1] == 3

    # Calculate padding
    pad11 = (target_size - axis_dims[0]) // 2
    pad12 = target_size - axis_dims[0] - pad11
    pad21 = (target_size - axis_dims[1]) // 2
    pad22 = target_size - axis_dims[1] - pad21

    padded_frame = np.pad(frame,
                          pad_width=((pad11,pad12),(pad21,pad22),(0,0)),
                          mode='constant',
                          constant_values=fill_color)    
    return padded_frame


########################## Aligner Functions ############################
### Frame-level functions with multiple data sources. E.g.,
### concatenating video frames, labeling a frame.                    ####
#########################################################################

def add_label_to_frame(
        aligner_item: Tuple[np.ndarray, Optional[str]],
        video_name: str = ""
) -> np.ndarray:
    """Adds label in white text box to top left corner of a frame.

    Args:
        frame (np.array): Frame to add label to.
        label (str): Label to add to the frame.
        video_name (str): Name of the video.
    """
    frame, label = aligner_item
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 0)  # Black text
    bg_color = (255, 255, 255)  # White background

    y_offset = 10

    if type(video_name) == str and video_name != "":
        # Calculate text size for the video name
        video_name_size, _ = cv2.getTextSize(video_name, font, font_scale, thickness)
        vnw, vnh = video_name_size

        # Draw a white rectangle for the video name
        video_top_left = (10, y_offset)
        video_bottom_right = (10 + vnw + 10, y_offset + vnh + 10)
        cv2.rectangle(frame, video_top_left, video_bottom_right, bg_color, -1)

        # Put the video name text on the rectangle
        video_text_position = (10 + 5, y_offset + vnh + 5)
        cv2.putText(frame, video_name, video_text_position, font, font_scale, color, thickness)

        # Update y_offset for the video name
        y_offset += vnh + 15  # Add spacing between label and video name

    if type(label) == str and label != "":
        # Calculate text size for the label
        label_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_width, label_height = label_size

        # Draw a white rectangle for the label
        label_top_left = (10, y_offset)
        label_bottom_right = (10 + label_width + 10, y_offset + label_height + 10)
        cv2.rectangle(frame, label_top_left, label_bottom_right, bg_color, -1)

        # Put the label text on the rectangle
        label_text_position = (10 + 5, y_offset + label_height + 5)
        cv2.putText(frame, label, label_text_position, font, font_scale, color, thickness)

    return frame


def concat_frames(aligner_item,
                  itot, jtot,
                  titles=None, padding=10, target_size=384
) -> np.ndarray:
    """Concatenate frames from multiple sources into a single frame.

    Fills in frames of `target_size` top to bottom, left to right (row first). The final
    frame has `itot` rows and `jtot` columns.
    """
    assert titles is None or len(titles) == len(aligner_item), "Titles must match number of sources."
    num_sources = len(aligner_item)

    # Calculate overall frame size
    padded_width = target_size + 2 * padding
    padded_height = target_size + 2 * padding

    grid = []
    for i in range(num_sources):

        # Crop and resize images
        frame = aligner_item[i]
        frame = pad_to_square(thumbnail(frame, target_size=target_size), target_size=target_size)

        # Add padding and title
        frame = np.pad(frame, ((padding, padding), (padding, padding), (0, 0)), constant_values=255)
        if titles is not None:
            frame = add_label_to_frame((frame, titles[i]))

        grid.append(frame)
    
    while len(grid) < itot * jtot:
        blank_img = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * 0
        grid.append(blank_img)

    rows = [np.hstack(grid[j * jtot:(j + 1) * jtot]) for j in range(itot)]
    return np.vstack(rows)


#################### Multiprocessing Functionality ####################


def input_reader_process(
        unprocessed_queue: Queue, aligner: Aligner, num_term_signals: int, error_event: Event
) -> None:
    """Reads from aligner and puts data into unprocessed queue."""
    idx = 0
    try:
        while True:
            if error_event.is_set():
                break
            n = next(aligner, None)
            if n is None:
                break

            ts, data_list = n
            unprocessed_queue.put((idx, ts, data_list))
            idx += 1
    except Exception as e:
        logging.error(f"Error in input_reader_process: {e}")
        error_event.set()
    finally:
        for _ in range(num_term_signals):
            unprocessed_queue.put(None)

def output_video_writer_process(processed_queue: Queue,
                                outpath: str, fps: float,
                                num_term_signals: int,
                                reverse_rgb: bool = False,
                                error_event: Event = None
) -> None:
    """Write frames from processed queue to video at outpath."""
    writer = None
    termination_signals = 0
    next_frame_idx = 0
    frame_buffer = {}

    pbar = tqdm(total=None, desc="Writing Frames", unit=" frames")

    try:
        while termination_signals < num_term_signals:
            if error_event.is_set():
                break
            item = processed_queue.get(timeout=10)
            if item is None:
                termination_signals += 1
                continue
            frame_idx, frame = item
            frame_buffer[frame_idx] = frame

            while next_frame_idx in frame_buffer:
                if writer is None:
                    frame_size = (frame.shape[1], frame.shape[0])
                    writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

                out_frame = frame_buffer.pop(next_frame_idx)
                if reverse_rgb:
                    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                writer.write(out_frame)
                next_frame_idx += 1

                pbar.update(1)
    except Exception as e:
        logging.error(f"Error in output_video_writer_process: {e}")
        error_event.set()
    finally:
        if writer is not None:
            writer.release()
        pbar.close()

def write_output_video(aligner: TimeAligner,
                       outpath: str, process_fn: Callable,
                       num_workers: int = 0,
                       reverse_rgb: bool = False
) -> None:
    """Handles multiprocessing. The main process reads from the aligner and
    puts data into the unprocessed queue. Multiple worker processes run the
    process_fn on the data in the queue and put the processed data into the
    processed queue. Finally, a child process writes the processed data to
    outpath.

    Args:
        aligner (TimeAligner): Aligner object that provides synchronized data. Assumes
            the first iterator is of video iterator type.
        outpath (str): Path to save the output video.
        process_fn (Callable): Function to apply to each item in the aligner iterator.
            Input: Aligner item, Output: processed frame.
        num_workers (int): Number of worker processes to use. If 0, uses all available CPUs.
        reverse_rgb (bool): Whether to reverse the RGB channels before writing to video.
            Done in writer process.
    """
    assert num_workers >= 0, "Number of workers must be non-negative."

    ensure_dir_exists(outpath)
    unprocessed_queue = Queue(maxsize=QUEUE_MAX_SIZE)
    processed_queue = Queue(maxsize=QUEUE_MAX_SIZE)
    error_event = Event()

    if num_workers == 0:
        num_workers = os.cpu_count()
    else:
        if num_workers > os.cpu_count():
            print(f"Specified {num_workers} workers too high, using {os.cpu_count()} workers instead.")
            num_workers = min(num_workers, os.cpu_count())

    def worker_loop():
        try:
            while True:
                if error_event.is_set():
                    break
                item = unprocessed_queue.get(timeout=10)
                if item is None:  # Done!
                    break
                frame_idx, _, data_list = item
                processed_frame = process_fn(data_list)
                processed_queue.put((frame_idx, processed_frame))
            processed_queue.put(None)
        except Exception as e:
            logging.error(f"Error in worker_loop: {e}")
            error_event.set()
            processed_queue.put(None)

    # Everyone else processes frames
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker_loop)
        p.start()
        processes.append(p)

    fps = 1.0 / aligner.sample_rate
    writer_proc = Process(target=output_video_writer_process,
                            args=(processed_queue, outpath, fps, num_workers, reverse_rgb, error_event))
    writer_proc.start()

    # Directly call reader process in the main process
    input_reader_process(unprocessed_queue, aligner, num_workers, error_event)

    for p in processes:
        p.join()
    writer_proc.join()

    if error_event.is_set():
        raise Exception


################################## Combined ######################################


def concat_pose_videos(pose2D_video_paths, pose3D_video_paths, outpath,
             titles=None, padding=10, target_size=384,
             num_workers=0
) -> None:
    """
    Visualize multiple pose models on the same video with padding and optional titles.

    Args:
        pose2D_video_paths (list): List of paths to 2D pose videos.
        pose3D_video_paths (list): List of paths to 3D pose videos.
        outpath (str): Path to save the output video with combined frames.
        titles (list): List of titles corresponding to each 2D-3D pair for labeling frames.
        padding (int): Amount of padding in pixels to add around each image.
    """
    assert len(pose2D_video_paths) > 0, "No 2D pose videos provided."
    assert len(pose2D_video_paths) == len(pose3D_video_paths)

    num_workers = os.cpu_count() if num_workers == 0 else num_workers
    N = len(pose2D_video_paths)
    titles = titles if titles else ["" for _ in range(N)]

    # Calculate grid dimensions
    itot = 1
    jtot = 2
    while itot * jtot < 2 * N:
        if itot == jtot:
            jtot += 2  # Increase columns first
        else:
            itot += 1

    vis = []
    for pose2D_path, pose3D_path in zip(pose2D_video_paths, pose3D_video_paths):
        assert os.path.exists(pose2D_path), f"2D pose video not found at {pose2D_path}"
        assert os.path.exists(pose3D_path), f"3D pose video not found at {pose3D_path}"
        vi2 = DecordVideoIterator(pose2D_path)
        vi3 = DecordVideoIterator(pose3D_path)
        vis.append(vi2)
        vis.append(vi3)
    aligner = TimeAligner(vis)

    pcf = partial(concat_frames,
                  itot=itot, jtot=jtot,
                  titles=titles, padding=padding,
                  target_size=target_size)
    write_output_video(aligner, outpath, pcf, num_workers, reverse_rgb=True)


def annotate_video_with_labels(videopath: str, labelspath: str, outpath: str,
                               num_workers: int = 0) -> None:
    """Annotate a video with labels from a CSV file."""
    num_workers = os.cpu_count() if num_workers == 0 else num_workers
    vn = os.path.basename(videopath)
    add_label_and_video_name = partial(add_label_to_frame, video_name=vn)
    aligner = TimeAligner(
        (DecordVideoIterator(videopath), PandasIterator(labelspath, data_col='MarkerNames', time_col='Time_s'))
    )
    write_output_video(aligner, outpath, add_label_and_video_name, num_workers, reverse_rgb=True)
