import cv2
import torch
import os
from transformers 


def annotate_image(image, bbox):
    """
    Draw bounding boxes, labels, and scores on an image.
    
    Parameters:
        image (numpy.ndarray): The input image in RGB format.
        bbox (list): List containing bounding box data with 'boxes', 'text_labels', and 'scores'.
    """
    
    # Convert image from RGB to BGR (for OpenCV display)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Unpack bounding box info
    for entry in bbox:
        boxes = entry['boxes'].cpu().numpy() if isinstance(entry['boxes'], torch.Tensor) else entry['boxes']
        labels = entry['text_labels']
        scores = entry['scores'].cpu().numpy() if isinstance(entry['scores'], torch.Tensor) else entry['scores']

        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label text
            text = f"{label}: {score:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + 10
            
            # Background for text
            cv2.rectangle(image_bgr, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0] + 2, text_y + 2), (0, 255, 0), -1)
            
            # Put text on image
            cv2.putText(image_bgr, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert back to RGB for correct display



def video_to_image(video_path, frame_number=0):
    """
    Extract a single frame from a video file.
    
    Parameters:
        video_path (str): Path to the video file.
        frame_number (int): The frame number to extract.
        
    Returns:
        numpy.ndarray: The extracted frame as an image.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from video {video_path}")
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

test_videos = [
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_brushing1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_brushing1_2.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_combing1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_combing1_2.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_deodrant1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_deodrant1_2.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_drinking1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_drinking1_2.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_face wash1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_face wash1_2.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_feeding1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_feeding1_2.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_glasses1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_glasses1_2.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_RTT left side1_1.avi",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_RTT left side1_2.avi",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_shelf right side1_1.mkv",
    "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_shelf right side1_1.mkv",
]

from tools.grounding_dino.inferencer import GroundingDINOInferencer
inferencer = GroundingDINOInferencer()

import pdb ; pdb.set_trace()
for video_path in test_videos:
    print(f"Processing video: {video_path}")
    
    # Extract a frame from the video
    try:
        image = 
        # image = video_to_image(video_path, frame_number=0)  # Change frame_number as needed
    except ValueError as e:
        print(e)
        continue
    
    prompts = ["person."]  # Example prompt, can be modified as needed
    results = inferencer._dino_detect(
        image=image,
        prompts=prompts,
        box_threshold=0.4,
        text_threshold=0.3
    )

    results = [
        {
            "boxes": [r[:4] for r in results["person."]],
            "text_labels": ["person"] * len(results["person."]),
            "scores": [r[4] for r in results["person."]],
        }
    ]

    # Annotate and save the image
    annotated_image = annotate_image(image, results)
    outpath = os.path.join("test_images_out", os.path.basename(video_path) + "_annotated.png")
    cv2.imwrite(outpath, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to {outpath}")

# [{'scores': tensor([0.7059, 0.5869, 0.4752], device='cuda:0'), 'boxes': tensor([[305.9848, 457.2027, 377.4766, 545.2834],
        # [618.2330, 484.0649, 714.2146, 571.7430],
        # [307.0463, 142.5136, 713.5226, 569.5648]], device='cuda:0'), 'text_labels': ['hand', 'hand', 'patient'], 'labels': ['hand', 'hand', 'patient']}]

# To process videos
# Take the highest scoring patient as the actual patient
# Which direction is the patient facing? 
