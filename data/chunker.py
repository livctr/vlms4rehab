import os
import cv2
import numpy as np
import decord
from decord import VideoReader, cpu

from tqdm import tqdm

import pandas as pd

from data.utils_strokerehab import VIDEO_DIR, CHUNKED_VIDEO_DIR

def resave_video(input_path: str,
                 output_path: str,
                 sampling_fps: int = 8) -> None:
    """
    Reads a video from `input_path` using decord, samples it at approximately 
    `sampling_fps` (by generating equally spaced frame indices), and saves it 
    as an MP4 video at 1 FPS at `output_path`.
    """
    # Open the video with decord.
    try:
        vr = VideoReader(input_path, ctx=cpu(0))
    except Exception as e:
        raise IOError(f"Cannot open video file: {input_path}. Error: {e}")

    # Get the original FPS and total number of frames.
    orig_fps = vr.get_avg_fps()
    num_frames = len(vr)
    if orig_fps <= 0 or num_frames == 0:
        raise ValueError("Original video FPS or frame count is invalid.")

    # Compute video duration in seconds.
    duration = num_frames / orig_fps
    
    # Calculate the total number of samples we want.
    num_samples = int(duration * sampling_fps)  # round down
    if num_samples < 1:
        raise ValueError("Calculated number of samples is less than 1.")

    # Generate equally spaced frame indices across the video duration.
    indices = [int(round(idx)) for idx in np.linspace(0, num_frames - 1, num_samples)]
    
    out_writer = None

    # Sample frames and write them to the output video.
    for idx in tqdm(indices):
        # Get frame in RGB format from decord
        frame = vr[idx].asnumpy()
        # Convert from RGB (decord) to BGR (cv2) format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Initialize the video writer if not already done
        if out_writer is None:
            # Prepare to write the output video using cv2.
            # Get frame dimensions from the first sampled frame.
            height, width = frame_bgr.shape[:2]
            # Ensure the output directory exists.
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Define codec and output FPS (1 FPS as required).
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_fps = 1
            out_writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

        # Write frame to output video
        out_writer.write(frame_bgr)

    out_writer.release()
    print(f"Resaved video to: {output_path}")


def resave_all_videos(in_dir: str, out_dir: str,
                      patients='all', activity='all', reps='all', filter_for_testset=False):
    df = pd.read_csv("data/csvs_txts_yamls/cleaned_metadata.csv")
    if patients != 'all':
        patients = patients.split(',')
        df = df[df['patient'].isin(patients)]
    if activity != 'all':
        activity = activity.split(',')
        df = df[df['activity'].isin(activity)]
    if reps != 'all':
        if reps != 'first':
            raise ValueError("Invalid value for reps. Must be 'all' or 'first'.")
        df = df.sort_values('id').groupby(['patient', 'activity']).agg('first').reset_index()
    if filter_for_testset:
        df = df[df['is_in_strokerehab_test_set']]
    
    for _, row in df.iterrows():
        input_path = os.path.join(in_dir, row['path_v'])
        output_path = os.path.join(out_dir, row['path_v'].split('.')[0]+'.mp4')
        resave_video(input_path, output_path)



if __name__ == "__main__":
    resave_all_videos(VIDEO_DIR, CHUNKED_VIDEO_DIR, patients='S0001', reps='first')
