import decord
import cv2
import os



# 100%|█████████▉| 5169/5187 [1:05:13<00:15,  1.15it/s]2025-03-31 12:20:57,716 - INFO - Hand detection error for /gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/S00026/S00026_face wash1_2.avi, frame 0: No hands detected in image


# Path to the video file
video_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/S00026/S00026_face wash1_2.avi"

# Initialize the video reader
vr = decord.VideoReader(video_path)

# Get the first frame (index 0)
frame = vr[0].asnumpy()

# Convert from RGB to BGR for OpenCV
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Save the frame
output_path = "scrap.png"
cv2.imwrite(output_path, frame_bgr)

print(f"Frame saved to {os.path.abspath(output_path)}")