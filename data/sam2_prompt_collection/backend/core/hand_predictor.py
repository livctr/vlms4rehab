import os
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from loguru import logger

class HandDetectionError(Exception):
    """Custom exception for when hands cannot be detected."""
    pass

def distance_to_box(point, box):
    # If point is inside box, return 0
    if (
        point[0] >= box[0]
        and point[0] <= box[2]
        and point[1] >= box[1]
        and point[1] <= box[3]
    ):
        return 0

    # Calculate distance to nearest edge
    dx = max(box[0] - point[0], 0, point[0] - box[2])
    dy = max(box[1] - point[1], 0, point[1] - box[3])
    return (dx**2 + dy**2) ** 0.5

class HandPredictor:
    def __init__(
        self,
        dino_model_id="IDEA-Research/grounding-dino-base",
        hand_iou_threshold=0.5,
        device=None
    ):
        """
        Initialize the HandPredictor with object detection model.

        Args:
            dino_model_id (str): The model ID for Grounding DINO zero-shot object detection
            hand_iou_threshold (float): IoU threshold for filtering hand bounding boxes
            device (str, optional): Device to run models on ('cuda' or 'cpu'). 
                                   If None, will use CUDA if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing HandPredictor on {self.device}")
        
        # Initialize Grounding DINO model and processor
        logger.info(f"Loading Grounding DINO model: {dino_model_id}")
        self.processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_id
        ).to(self.device)

        self.hand_iou_threshold = hand_iou_threshold
    
    def _get_bboxes(self, image, query):
        inputs = self.processor(
            images=image,
            text=f"{query}.",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        dino_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.shape[:-1]],
        )

        bboxes = []
        if dino_results and len(dino_results[0]["text_labels"]) > 0:
            # Sort by confidence score
            indices = [
                i for i, label in enumerate(dino_results[0]["text_labels"]) if label == query
            ]
            indices.sort(key=lambda i: dino_results[0]["scores"][i].item(), reverse=True)
            bboxes = [
                dino_results[0]["boxes"][i].cpu().numpy().astype(int).tolist() for i in indices
            ]
        return bboxes

    def detect_hands(self, image):
        """
        Detect all hands in the image using Grounding DINO.
        
        Args:
            image (numpy.ndarray): (H, W, 3) numpy array in RGB format.
            
        Returns:
            dict: A dictionary containing:
                - 'hand_bboxes': List of [x1, y1, x2, y2] bounding boxes for detected hands
        """
        hand_bboxes = self._get_bboxes(image, "hand")
        return {"hand_bboxes": hand_bboxes}
        

    def human_itl_detect_hands(self, human_input):
        """
        Process human input for hand detection.
        
        Args:
            human_input (dict): Dictionary containing:
                - "hand_bboxes": List of [x1, y1, x2, y2] bounding boxes
                - "left_hand_idx": Index of selected left hand box (-1 if none)
                - "right_hand_idx": Index of selected right hand box (-1 if none)
                
        Returns:
            dict: Dictionary with left and right hand bounding boxes
        """
        for key in ["hand_bboxes", "left_hand_idx", "right_hand_idx"]:
            if key not in human_input:
                raise HandDetectionError(f"Missing key from human input: {key}")
        
        left_idx = human_input["left_hand_idx"]
        right_idx = human_input["right_hand_idx"]
        
        # Validate indices
        if left_idx >= len(human_input["hand_bboxes"]) or right_idx >= len(human_input["hand_bboxes"]):
            raise HandDetectionError("Invalid hand index")
        
        # Get selected boxes
        left_hand = human_input["hand_bboxes"][left_idx] if left_idx >= 0 else None
        right_hand = human_input["hand_bboxes"][right_idx] if right_idx >= 0 else None

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "hand_bboxes": human_input["hand_bboxes"]
        }
