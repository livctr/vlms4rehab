from typing import Dict, Any, Optional, Tuple
import base64
import io
from PIL import Image

import numpy as np

from tqdm import tqdm

from data.pipeline.backend.core.hand_predictor import HandPredictor, HandDetectionError, PatientDetectionError
from data.pipeline.backend.core.human_input_data_manager import HumanInputDataManager

class CollectionAPI:
    def __init__(self,
                 hand_predictor: HandPredictor,
                 data_manager: HumanInputDataManager):
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
            data = self.data_manager.prev()
        elif position == "current":
            data = self.data_manager.current()
        elif position == "next":
            data = self.data_manager.next()
        else:
            raise ValueError(f"Invalid position: {position}")

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
        self.data_manager.annotate_cur(human_input["path_v"],
                                       human_input["frame_idx"],
                                       result)
        return {"success": True}
    
    def save(self):
        self.data_manager.save()
        return {"success": True}


class AutoAPI:
    def __init__(self,
                 hand_predictor: HandPredictor,
                 data_manager: HumanInputDataManager,
                 ):
        """
        Initialize the CollectionAPI with data manager and hand predictor.
        """
        self.hand_predictor = hand_predictor
        self.data_manager = data_manager
        self.num_success = 0

    def run(self):
        data = self.data_manager.current()
        total_frames = data['num_frames_needed']
        for _ in tqdm(range(total_frames)):
            self.predict(data)
            data = self.data_manager.next()
        self.data_manager.save()
        print(f"{self.num_success} frames successfully annotated out of {total_frames} total frames.")

    def predict(self, data):

        path_v = data["path_v"]
        frame_idx = data["frame_idx"]
        frame = data["frame"]

        try:
            result = self.hand_predictor.auto_detect_patient_and_hands(
                frame,
                hand=data['handedness']
            )
            self.data_manager.annotate_cur(path_v, frame_idx, result)
            self.num_success += 1
            print(f"Auto detection successful for {path_v}, frame {frame_idx}")
        except HandDetectionError as e:
            print(f"Hand detection error for {path_v}, frame {frame_idx}: {e}")
        except PatientDetectionError as e:
            print(f"Patient detection error for {path_v}, frame {frame_idx}: {e}")
