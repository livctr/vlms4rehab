import os
import subprocess

import numpy as np
import pandas as pd
from data.utils_strokerehab import DataPaths
from data.viz import annotate_video_with_labels

import cv2



def get_first_frame(row, outpath):
    video_path = row['path_v']
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Check if a frame was successfully retrieved
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read the first frame from the video")
    
    # Write the frame to the output path
    # Ensure the directory exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    cv2.imwrite(outpath, frame)
    
    # Release the video capture object
    cap.release()


def show_video_label_pair(i, row):
    video_name = os.path.basename(row['path_v']).split('.')[0]
    outpath = os.path.join(DataPaths.VERIFICATION_PATH, video_name + '_tmp.mp4')
    encoded_outpath = os.path.join(DataPaths.VERIFICATION_PATH, video_name + '.mp4')

    try:

        if os.path.exists(encoded_outpath):
            print(f"Video {video_name} (index {i}) is already processed")
            return
        print(f"Video {video_name} (index {i}) is being processed")

        annotate_video_with_labels(row['path_v'], row['path_l'], outpath, num_workers=8)

        encode_command = [
            'ffmpeg', '-n', '-i', outpath, 
            '-c:v', 'libx264', 
            '-preset', 'fast', 
            '-crf', '23', 
            '-c:a', 'aac', 
            '-strict', 'experimental', 
            encoded_outpath
        ]
        subprocess.run(encode_command, stdout=subprocess.DEVNULL, check=True)
        rm_command = ['rm', outpath]
        subprocess.run(rm_command, stdout=subprocess.DEVNULL, check=True)
    
    except Exception as e:
        print(f"Error processing video {i}: {video_name}. {e}")


if __name__ == "__main__":

    df = pd.read_csv(DataPaths.METADATA_PATH)
    np.random.seed(42)
    # sampled_df = df[df['path_l']=='/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/S00027/S00027_RTT left side2_1.csv']
    paths_l = [
        # '/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/S0001/S0001_face wash1_2.csv',
        # '/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/S0001/S0001_glasses1_1.csv',
    ]
    # sampled_df = df[df['path_l'].isin(paths_l)]

    sampled_df = df.sample(n=10).reset_index(drop=True)
    print(sampled_df)
    # sampled_df.to_csv("/gpfs/data/schambralab/quantitativeRehabilitation/__data/metadata/metadata_video_n_labels.csv")

    for i, row in sampled_df.iterrows():
        # show_video_label_pair(i, row)
        video_path = row['path_v']
        base_name = os.path.basename(video_path).split('.')[0]
        get_first_frame(row, os.path.join("/gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/victor/explore/grounding_dino", base_name + '.jpg'))


    # rclone_command = [
    #     "rclone", "copy",
    #     DataPaths.VERIFICATION_PATH,
    #     "nyu-drive:carlos-group/strokerehab/strokerehab-videos/videos_n_labels"
    # ]
    # subprocess.run(rclone_command, check=True)
