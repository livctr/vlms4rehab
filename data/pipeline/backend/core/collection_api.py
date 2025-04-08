from typing import Dict, Any, Optional, Tuple
import base64
import io
from PIL import Image
import logging

import numpy as np

from tqdm import tqdm

from data.pipeline.backend.core.hand_predictor import HandPredictor, HandDetectionError, PatientDetectionError
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
    
    def select_hand_bbox(self, human_input):
        result = self.hand_predictor.human_itl_detect_patient_and_hands(
            human_input=human_input,
            hand=human_input["handedness"]
        )
        self.data_manager.annotate_cur(result,
                                       human_input["path_v"],
                                       human_input["frame_idx"])
        return {"success": True}
    
    def save(self):
        self.data_manager.save()
        return {"success": True}


class AutoAPI:
    def __init__(self,
                 hand_predictor: HandPredictor,
                 data_manager: DataManager,
                 ):
        """
        Initialize the CollectionAPI with data manager and hand predictor.
        """
        self.hand_predictor = hand_predictor
        self.data_manager = data_manager
        self.num_success = 0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        tf = self.data_manager.total_frames_needed
        self.logger.info(f"Starting auto annotation for {tf} frames.")
        for i in tqdm(range(tf), desc="Auto annotation progress", unit="frame"):
            data = self.data_manager.current()
            self.predict(data)
            self.data_manager.next()

            if i % 100 == 0:
                self.data_manager.save()

        self.data_manager.save()
        self.logger.info(f"{self.num_success} frames successfully annotated out of {tf} total frames.")

    def predict(self, data):
        path_v = data["path_v"]
        frame_idx = data["frame_idx"]
        frame = data["frame"]
        if frame is None:
            self.logger.info(f"Frame not found for {path_v}, frame {frame_idx}. "
                             "Setting data to empty dictionary.")
            self.data_manager.annotate_cur({}, path_v, frame_idx)
            self.num_success += 1
            return

        try:
            result = self.hand_predictor.auto_detect_patient_and_hands(
                frame,
                hand=data['handedness']
            )
            self.data_manager.annotate_cur(result, path_v, frame_idx)
            self.num_success += 1
            self.logger.info(f"Auto detection successful for {path_v}, frame {frame_idx}")
        except HandDetectionError as e:
            self.logger.info(f"Hand detection error for {path_v}, frame {frame_idx}: {e}")
        except PatientDetectionError as e:
            self.logger.info(f"Patient detection error for {path_v}, frame {frame_idx}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error for {path_v}, frame {frame_idx}: {e}")
