import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Dict, Any, Optional, Tuple
import time

from data.visualization.data_streamer import DecordVideoStreamer
from tools.sort import Sort
from tools.sort.sort import iou


class GroundingDINOInferencer:
    """
    A class to perform inference with GroundingDINO on video files.
    """
    def __init__(self,
                 model_id: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda"):
        """
        Initializes the processor and model.

        Args:
            model_id (str): The model ID from Hugging Face Hub.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device = "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model_id = model_id

    def _enlarge_and_clip(self, box, shape, reapply_scale = 1.2):
        """Scale box by reapply_scale about its center and clip to image."""
        x1,y1,x2,y2 = box
        w, h = x2-x1, y2-y1
        cx, cy = x1 + w/2, y1 + h/2
        new_w, new_h = w*reapply_scale, h*reapply_scale
        x1n = max(0, cx - new_w/2)
        y1n = max(0, cy - new_h/2)
        x2n = min(shape[1], cx + new_w/2)
        y2n = min(shape[0], cy + new_h/2)
        return int(x1n), int(y1n), int(x2n), int(y2n)

    def _dino_detect(
        self,
        image: np.ndarray,
        prompts: List[str],
        box_threshold: float,
        text_threshold: float,
        crop_box: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Runs GroundingDINO on either the full image or a cropped sub-region.
        
        Args:
            image (np.ndarray): Full original image (H × W × C).
            prompts (List[str]): List of text prompts to detect.
            box_threshold (float): Box confidence threshold.
            text_threshold (float): Text similarity threshold.
            crop_box (List[int], optional): [x1, y1, x2, y2] in original-image coords;
                if provided, only that sub-region is cropped and passed to the model.

        Returns:
            List[Dict]: One entry per detected box:
                {
                "text_label": <int label>,
                "box": [x1, y1, x2, y2, score]
                }
                where coordinates are relative to the original image.
        """
        # If requested, crop the image
        x_off = y_off = 0
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            crop = image[y1:y2, x1:x2]
            x_off, y_off = x1, y1
        else:
            crop = image

        # Prepare target size tensor for the (possibly cropped) image
        # (batch of one: height, width)
        target_sizes = torch.tensor([crop.shape[:2]], device=self.device)

        frame_results: Dict[List] = {p: [] for p in prompts}

        for p in prompts:
            inputs = self.processor(
                images=crop,
                text=p,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            post = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes
            )[0]

            # Map each box back to original coords and collect
            for score, _, box in zip(post["scores"], post["text_labels"], post["boxes"]):
                x1c, y1c, x2c, y2c = box.tolist()
                frame_results[p].append([
                    round(x1c + x_off, 2),
                    round(y1c + y_off, 2),
                    round(x2c + x_off, 2),
                    round(y2c + y_off, 2),
                    round(score.item(), 3)
                ])

        return frame_results

    def process_video(self,
                      video_path: str,
                      texts: List[str],
                      box_threshold: float = 0.4,
                      text_threshold: float = 0.3,
                      max_age_s: float = 10.0,
                      min_hit_s: float = 0.2,
                      sample_fps: Optional[float] = None,
                      do_reapply: bool = True,
                      min_crop_size: int = 25,
                      iou_reapply: float = 0.3,
                      reapply_scale: float = 1.2
                  ) -> Dict[str, Any]:
        """
        Performs object detection on video frames using text-based prompts.

        Args:
            video_path (str): Path to the input video file
            text (str): List of objects to detect in text. Each object should be its
                own string, e.g. ["patient", "hand", "bottle"].
            box_threshold (float): Confidence threshold for box detection
            text_threshold (float): Confidence threshold for text matching
            iou_reapply (float): if GroundingDINO does not predict a box with an IOU higher
                than iou_reapply w.r.t. the predicted box from the previous frame(s),
                GroundingDINO will be applied again to the crop around the predicted box.
                Essentially, we will "look closer" for missed detections at the predicted
                locations.
            reapply_scale (float): Scale factor to enlarge boxes before reapplying IOU.
            sample_fps (float, optional): Frame rate to sample video. Uses video's FPS if None
        Returns:
            Dict[str, Any]: Detection results containing metadata, timestamps and bounding boxes
        """
        streamer = DecordVideoStreamer(video_path, read_from_cpu_id=0)
        sample_fps = sample_fps if sample_fps is not None else streamer.video_fps
        sample_rate = 1. / sample_fps
        streamer.sample_rate = sample_rate
        video_metadata = streamer.metadata
        max_age = int(max_age_s / sample_rate)
        min_hits = int(min_hit_s / sample_rate)


        prompts = [p if p.endswith('.') else p + '.' for p in texts]
        labels = set(prompts)  # the prompts are the labels
        trackers = {label: Sort(max_age=max_age, min_hits=min_hits) for label in labels}

        video_results = {
            "metadata": {
                "video_path": video_path,
                "model_id": self.model_id,
                "box_format": "[x_min, y_min, x_max, y_max, score, id]",
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                **video_metadata
            },
            "Time_s": [],
            "Boxes": []
        }

        from tqdm import tqdm
        for ts, frame in tqdm(streamer):

            video_results['Time_s'].append(round(ts, 3))

            frame_detections = []

            # 1) Full-frame detection
            detections = self._dino_detect(
                image=frame,
                prompts=prompts,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            for label, tracker in trackers.items():

                # Get predicted positions from Kalman
                preds = []
                for trk in tracker.trackers:
                    # convert kf.x → bbox (center to corners)
                    preds.append(trk.get_prediction()[0])

                full_boxes = {label: detections[label] for label in labels}

                # Get detections from crop
                for pb in preds:
                    # Check overlap vs full-frame detections
                    if do_reapply and (
                        len(full_boxes) == 0 or 
                        max(iou(pb, fb) for fb in full_boxes[label])
                    ) < iou_reapply:
                        # Crop & rerun DINO on that region
                        x1, y1, x2, y2 = self._enlarge_and_clip(pb, frame.shape, reapply_scale)
                        valid_crop = (
                            x2 > x1 + min_crop_size and y2 > y1 + min_crop_size
                        )
                        if not valid_crop:
                            continue
                        print(x1, y1, x2, y2, valid_crop)
                        crop_dets = self._dino_detect(
                            image=frame,
                            prompts=[label],
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                            crop_box=[x1, y1, x2, y2]
                        )
                        # Project back to full frame
                        for nd in crop_dets[label]:
                            bx_full = [
                                round(nd[0], 2),
                                round(nd[1], 2),
                                round(nd[2], 2),
                                round(nd[3], 2),
                                nd[4]
                            ]
                            detections[label].append(bx_full)
                
                # Update trackers with the detections from crop
                for identified_det in tracker.update(
                    np.array(detections[label]),
                    include_score=True
                ):
                    x1, y1, x2, y2, score, id_ = identified_det.tolist()
                    box = [x1, y1, x2, y2]
                    frame_detections.append({
                        "text_label": label,
                        "box": [round(coord, 2) for coord in box],
                        "score": round(score, 3),
                        "id": int(id_)
                    })
            
            video_results["Boxes"].append(frame_detections)

        return video_results


if __name__ == "__main__":
    # import pdb ; pdb.set_trace()
    import json
    import os

    # 1. Create a dummy video for the demonstration
    test_videos = [
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_brushing1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_brushing1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_combing1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_combing1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_deodrant1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_deodrant1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_drinking1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_drinking1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_face wash1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_face wash1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_feeding1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_feeding1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_glasses1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_glasses1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_RTT left side1_1.avi",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_RTT left side1_2.avi",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_shelf right side1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_shelf right side1_2.mkv",
        "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv"
    ]

    # 2. Instantiate the inferencer
    # Use 'cpu' if you don't have a CUDA-enabled GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inferencer = GroundingDINOInferencer(device=device)

    # 3. Define the objects to detect in the video
    labels_to_find = ["hand"]

    # 4. Run inference on all test videos
    print("\nStarting video inference...")
    for video_path in test_videos:
        print(f"\nProcessing {video_path}")
        detection_results = inferencer.process_video(
            video_path=video_path,
            texts=labels_to_find,
        )

        # Create output filename
        video_name = video_path.split('/')[-1].split('.')[0]
        output_path = f"test_videos_out/gd_with_cropping_{video_name}.json"

        # Ensure output directory exists
        os.makedirs("test_videos_out", exist_ok=True)

        # Save results
        with open(output_path, "w") as f:
            json.dump(detection_results, f, indent=2)
        print(f"Results saved to {output_path}")
