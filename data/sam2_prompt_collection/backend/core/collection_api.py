from typing import Dict, Any, Optional, Tuple
import base64
import io
from PIL import Image
import logging

import numpy as np

from tqdm import tqdm

from data.pipeline.backend.core.hand_predictor import HandPredictor, HandDetectionError
from data.pipeline.backend.core.data_manager import DataManager

class CollectionAPI:
    def __init__(self,
                 hand_predictor: HandPredictor,
                 data_manager: DataManager):
        """
        Initialize the CollectionAPI with data manager and hand predictor.
        """
        self.hand_predictor = hand_predictor
        self.data_manager = data_manager
    
    def get_data(self, position: str) -> Dict[str, Any]:
        """
        Get image data and try auto-detection.
        
        Args:
            position: 'previous', 'current' or 'next'
        
        Returns:
            Dictionary with image data and detection results
        """
        # Get the image and its associated data
        if position == "previous":
            self.data_manager.prev()
            data = self.data_manager.current()
        elif position == "next":
            self.data_manager.next()
            data = self.data_manager.current()
        elif position == "current":
            data = self.data_manager.current()
        else:
            raise ValueError(f"Invalid position: {position}")
    
        frame = data.get("frame")  # "frame" always in dictionary, but may be None
        if frame is None:
            # Mark that the frame has no info available
            self.data_manager.annotate_cur({}, data["path_v"], data["frame_idx"])

        # Get hand detection data
        hands_data = self.hand_predictor.detect_hands(data['frame'])
        img_pil = Image.fromarray(data['frame'])
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')
        buffer = buffer.getvalue()
        data['frame'] = base64.b64encode(buffer).decode('utf-8')

        return {"success": True, **data, **hands_data}
    
    def select_hands(self, human_input):
        """
        Process human input for hand selection.
        
        Args:
            human_input (dict): Dictionary containing:
                - "hand_bboxes": List of [x1, y1, x2, y2] bounding boxes
                - "left_hand_idx": Index of selected left hand box (-1 if none)
                - "right_hand_idx": Index of selected right hand box (-1 if none)
        """
        result = self.hand_predictor.human_itl_detect_hands(human_input)
        self.data_manager.annotate_cur(result,
                                     human_input["path_v"],
                                     human_input["frame_idx"])
        return {"success": True}
    
    def save(self):
        self.data_manager.save()
        return {"success": True}
