import os
import subprocess

import numpy as np
import pandas as pd
from data.utils_strokerehab import DataPaths
from visualization.viz import annotate_video_with_labels

if __name__ == "__main__":

    df = pd.read_csv(DataPaths.METADATA_PATH)
    np.random.seed(42)
    sampled_df = df[df['path_l']=='/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/S00027/S00027_RTT left side2_1.csv']
    # sampled_df = df.sample(n=100).reset_index(drop=True)
    # sampled_df.to_csv("/gpfs/data/schambralab/quantitativeRehabilitation/__data/metadata/metadata_video_n_labels.csv")

    for i, row in sampled_df.iterrows():

        video_name = os.path.basename(row['path_v']).split('.')[0]
        outpath = os.path.join(DataPaths.VERIFICATION_PATH, video_name + '_tmp.mp4')
        encoded_outpath = os.path.join(DataPaths.VERIFICATION_PATH, video_name + '.mp4')

        try:

            if os.path.exists(encoded_outpath):
                print(f"Video {video_name} (index {i}) is already processed")
                continue
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
            continue

    # rclone_command = [
    #     "rclone", "copy",
    #     DataPaths.VERIFICATION_PATH,
    #     "nyu-drive:carlos-group/strokerehab/strokerehab-videos/videos_n_labels"
    # ]
    # subprocess.run(rclone_command, check=True)
