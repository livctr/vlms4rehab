import copy
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from data.visualization.data_streamer import DecordVideoStreamer
from tools.sort import Sort
from tools.sort.sort import iou
import cv2


class DinoSortTracker:
    def __init__(
        self,
        model_id="IDEA-Research/grounding-dino-base",
        device="cuda",
        iou_reapply: float = 0.3,
        reapply_scale: float = 1.2,
        max_age_s: float = 2.0,
        min_hits_s: float = 0.5,
    ):
        # load GroundingDINO
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        # tracking config
        self.iou_reapply = iou_reapply
        self.reapply_scale = reapply_scale
        self.trackers = {}  # label -> Sort instance
        self.max_age_s = max_age_s
        self.min_hits_s = min_hits_s

    def _dino_detect(self, image, texts, box_threshold, text_threshold):
        """
        Run GroundingDINO on a single image (either full frame or a crop).
        Returns a list of dicts: {"text_label", "box":[x1,y1,x2,y2,score]}
        """
        dets = []
        for text in texts:
            if not text.endswith("."):
                text += "."

            inputs = self.processor(
                images=image, text=text,
                padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                out,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=torch.tensor([image.shape[:2]]).to(self.device)
            )[0]
            for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
                dets.append({
                    "text_label": label,
                    "box": [round(c,2) for c in box.tolist()] + [round(score.item(),3)]
                })
        return dets

    def _enlarge_and_clip(self, box, shape):
        """Scale box by self.reapply_scale about its center and clip to image."""
        x1,y1,x2,y2 = box
        w, h = x2-x1, y2-y1
        cx, cy = x1 + w/2, y1 + h/2
        new_w, new_h = w*self.reapply_scale, h*self.reapply_scale
        x1n = max(0, cx - new_w/2)
        y1n = max(0, cy - new_h/2)
        x2n = min(shape[1], cx + new_w/2)
        y2n = min(shape[0], cy + new_h/2)
        return int(x1n), int(y1n), int(x2n), int(y2n)

    def process_video(
        self,
        video_path: str,
        texts: list[str],
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        max_age_s: float = 2,
        min_hits_s: float = 0.5,
        sample_fps: float = None,
    ):
        # import pdb ; pdb.set_trace()
        streamer = DecordVideoStreamer(video_path, read_from_cpu_id=0)
        sample_fps = sample_fps if sample_fps is not None else streamer.video_fps
        sample_rate = 1. / sample_fps
        max_age = int(max_age_s / sample_rate)
        min_hits = int(min_hits_s / sample_rate)

        streamer.sample_rate = sample_rate
        video_metadata = streamer.metadata
        video_results = {
            "metadata": {
                "video_path": video_path,
                "fps": video_metadata["fps"],
                "model_id": self.model_id,
                "box_format": "[x_min, y_min, x_max, y_max, score, id]",
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                **video_metadata
            },
            "Time_s": [],
            "Boxes": []
        }

        frame_idx = 0
        from tqdm import tqdm
        for ts, frame in tqdm(streamer):
            frame_idx += 1

            # import pdb ; pdb.set_trace()

            # 1) Full-frame detection
            dets_full = self._dino_detect(frame, texts, box_threshold, text_threshold)

            # On first frame: init one Sort per label
            if frame_idx == 1:
                labels = {d["text_label"] for d in dets_full}
                max_age = int(self.max_age_s / sample_rate)
                min_hits = int(self.min_hits_s / sample_rate)
                for label in labels:
                    self.trackers[label] = Sort(max_age=max_age, min_hits=min_hits)

            # group full-frame dets by label
            dets_by_label = {}
            for d in dets_full:
                dets_by_label.setdefault(d["text_label"], []).append(d.copy())

            # 2) For each existing track, see if we need to re-detect
            for label, tracker in self.trackers.items():
                # get predicted positions from Kalman (current state, not updating)
                preds = []
                for trk in tracker.trackers:
                    # convert kf.x → bbox (center to corners)
                    state = trk.kf.x  # [x,y,s,r,...]
                    px = float(state[0]); py = float(state[1])
                    s = float(state[2]); r = float(state[3])
                    w = np.sqrt(s*r); h = s / w
                    x1, y1 = px - w/2, py - h/2
                    x2, y2 = px + w/2, py + h/2
                    preds.append((x1,y1,x2,y2))

                # check overlap vs full-frame dets
                full_boxes = [d["box"][:4] for d in dets_by_label.get(label,[])]
                for pb in preds:
                    if len(full_boxes)==0 or max(iou(pb,fb) for fb in full_boxes) < self.iou_reapply:
                        # crop & rerun DINO on that region
                        x1,y1,x2,y2 = self._enlarge_and_clip(pb, frame.shape)
                        crop = frame[y1:y2, x1:x2]
                        new_dets = self._dino_detect(crop, [label], box_threshold, text_threshold)
                        # project back to full frame
                        for nd in new_dets:
                            bx = nd["box"]  # [x1',y1',x2',y2',score]
                            bx_full = [
                                round(bx[0]+x1,2),
                                round(bx[1]+y1,2),
                                round(bx[2]+x1,2),
                                round(bx[3]+y1,2),
                                bx[4]
                            ]
                            dets_by_label.setdefault(label,[]).append({
                                "text_label": label,
                                "box": bx_full
                            })

            # 3) Now update trackers with merged detections
            frame_tracked = []
            for label, tracker in self.trackers.items():
                arr = np.array([d["box"] for d in dets_by_label.get(label,[])])
                # shape (N,5) even if empty
                tracked = tracker.update(arr, include_score=True)
                # tracked: (M,6) → x1,y1,x2,y2,score,id
                for x1,y1,x2,y2,score,tid in tracked.tolist():
                    frame_tracked.append({
                        "text_label": label,
                        "box": [round(x1,2),round(y1,2),round(x2,2),round(y2,2)],
                        "score": round(score,3),
                        "id": int(tid),
                    })

            # yield or store per-frame results
            video_results["Time_s"].append(round(ts, 3))
            video_results["Boxes"].append(frame_tracked)
        
        return video_results

if __name__ == "__main__":
    import json

    # import pdb ; pdb.set_trace()

    # 1. Create a dummy video for the demonstration
    dummy_video_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv"

    # 2. Instantiate the inferencer
    # Use 'cpu' if you don't have a CUDA-enabled GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inferencer = DinoSortTracker(device=device)

    # 3. Define the objects to detect in the video
    labels_to_find = ["person", "hand"]

    # 4. Run inference
    print("\nStarting video inference...")
    detection_results = inferencer.process_video(
        video_path=dummy_video_path,
        texts=labels_to_find,
    )

    # 5. Print the JSON-serializable output
    print("\n--- Inference Results ---")
    print(json.dumps(detection_results, indent=2))

    # Optional: Save results to a file
    with open("detection_results.json", "w") as f:
        json.dump(detection_results, f, indent=2)
    print("\nResults saved to detection_results.json")