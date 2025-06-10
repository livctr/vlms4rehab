import os
import copy
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from tools.grounding_dino.inferencer import GroundingDINOInferencer


def video_to_image(video_path: str, frame_number: int = 0) -> np.ndarray:
    """
    Extract a single frame from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from video {video_path}")

    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compute_box_similarities(
    image: np.ndarray,
    boxes: list,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    text_prompt: str = "the patient in main focus, as highlighted by a green bounding box" 
) -> list:
    """
    For each bounding box, draw it in green on a copy of the image,
    then compute CLIP cosine similarity with the given text.
    Returns a list of similarity scores.
    """
    sims = []
    for box in boxes:
        temp_img = image.copy()
        x1, y1, x2, y2 = map(int, box)
        # Draw green bounding box
        temp_bgr = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(temp_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        temp_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)

        # Prepare PIL image for CLIP
        pil_img = Image.fromarray(temp_rgb)
        inputs = clip_processor(text=[text_prompt], images=pil_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
        img_embed = outputs.image_embeds
        txt_embed = outputs.text_embeds
        # normalize embeddings
        img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
        txt_embed = txt_embed / txt_embed.norm(p=2, dim=-1, keepdim=True)
        sim = torch.matmul(img_embed, txt_embed.T).item()
        sims.append(sim)
    return sims


def annotate_with_clip_scores(
    image: np.ndarray,
    boxes: list,
    sims: list
) -> np.ndarray:
    """
    Annotate the image with each box's CLIP similarity score.
    The highest-scoring box is drawn with a thicker border.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # determine which box has highest similarity
    best_idx = int(np.argmax(sims))

    for i, (box, score) in enumerate(zip(boxes, sims)):
        x1, y1, x2, y2 = map(int, box)
        thickness = 5 if i == best_idx else 2
        # draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        # annotate score
        text = f"{score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx, ty = x1, y1 - 10 if y1 - 10 > 10 else y1 + 10
        # background rectangle for readability
        cv2.rectangle(
            image_bgr,
            (tx, ty - text_size[1] - 2),
            (tx + text_size[0] + 2, ty + 2),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            image_bgr,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import pdb ; pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize GroundingDINO and CLIP

    inferencer = GroundingDINOInferencer()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    clip_model.eval()

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

    for video_path in test_videos:
        print(f"Processing video: {video_path}")
        try:
            image = video_to_image(video_path, frame_number=0)
        except ValueError as e:
            print(e)
            continue

        # Run GroundingDINO detection
        raw_results = inferencer._dino_detect(
            image=image,
            prompts=["person."],
            box_threshold=0.4,
            text_threshold=0.3
        )

        dets = raw_results.get("person.", [])
        if not dets:
            print("No detections found.")
            continue

        # Extract boxes
        boxes = [r[:4] for r in dets]

        # Compute CLIP similarity scores for each box
        sims = compute_box_similarities(image, boxes, clip_model, clip_processor, device)

        # Annotate original frame with CLIP scores
        annotated = annotate_with_clip_scores(image, boxes, sims)

        # Save output image
        out_dir = "test_images_out"
        os.makedirs(out_dir, exist_ok=True)
        outpath = os.path.join(out_dir, os.path.basename(video_path) + "_clip_annotated.png")
        cv2.imwrite(outpath, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Saved annotated image to {outpath}")
