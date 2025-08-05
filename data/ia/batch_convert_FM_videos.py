"""
Writes the videos to a `.mp4` for better viewing in external players.
"""

import re
import os

from data.utils_strokerehab import DataPaths, AFFECTED_PATIENTS


def convert_video_file(input_path):
    """
    Convert a single video file to FM format using ffmpeg.
    
    :param input_path: Path to the input video file.
    """
    if not os.path.exists(input_path):
        print(f"Input file {input_path} does not exist.")
        return

    # Construct the output path to mimic the input directory structure
    # Get the path relative to DataPaths.RAW_VIDEO_DIR
    relative_path = os.path.relpath(input_path, DataPaths.RAW_VIDEO_DIR)
    
    # Construct the full output path by joining DataPaths.IA_RAW_VIDEO_DIR with the relative path
    # and ensuring the extension is .mp4
    base_name = os.path.splitext(relative_path)[0]
    output_path = os.path.join(DataPaths.IA_RAW_VIDEO_DIR, f"{base_name}.mp4")

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping conversion.")
        return
    
    # Create the output directory (and any necessary parent directories)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = f"ffmpeg -i \"{input_path}\" -r 30 -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \"{output_path}\""
    os.system(command)
    print(f"Converted {input_path} to {output_path} using ffmpeg.")


def get_all_video_files(video_path_regex=None):
    video_files = []
    for root, _, files in os.walk(DataPaths.RAW_VIDEO_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            
            if file.endswith(('.mp4', '.avi', '.mov')) and "_FM" in file:
                if video_path_regex is None or re.search(video_path_regex, full_path):
                    video_files.append(full_path)
    return sorted(video_files)


if __name__ == "__main__":

    patient_ids = AFFECTED_PATIENTS.replace('"', '').split(',')  # replace is probably unnecessary
    patient_ids = [pid.strip() for pid in patient_ids]

    patient_pattern = r"({})[\\/]" .format("|".join(patient_ids))

    video_files = get_all_video_files(video_path_regex=patient_pattern)

    for video_file in video_files:
        print(f"Processing video file: {video_file}")
        convert_video_file(video_file)
