from ultralytics import YOLO
import os
import numpy as np

# Ground dino
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os

from loguru import logger

class PatientDetectionError(Exception):
    """Custom exception for when no patient is detected, multiple patients, etc."""
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
        pose_model_id="yolo11x-pose.pt",
        hand_iou_threshold=0.5,
        confidence_threshold=0.5,
        coco_kpts_threshold=3,
        device=None
    ):
        """
        Initialize the HandPredictor with object detection and pose estimation models.
        
        Args:
            dino_model_id (str): The model ID for Grounding DINO zero-shot object detection
            pose_model_id (str): The model path or name for YOLO pose estimation
            hand_iou_threshold (float): IoU threshold for filtering hand bounding boxes
            confidence_threshold (float): Confidence threshold for keypoints
            coco_kpts_threshold (int): Minimum number of COCO keypoints inside patient bbox
                to be considered the same patient
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
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLO pose estimation model
        logger.info(f"Loading YOLO pose model: {pose_model_id}")
        self.pose_model = YOLO(pose_model_id)
    
    def _filter_hand_bboxes(self, hand_bboxes, patient_bbox):
        """2 step filter: inside patient bounding box, NMS suppression.
        
        Assumes that hand_bboxes are already sorted by score."""
        # Filter hand boxes to be inside the patient bounding box
        inside_patient_bboxes = []
        for bbox in hand_bboxes:
            if (
                bbox[0] >= patient_bbox[0]
                and bbox[2] <= patient_bbox[2]
                and bbox[1] >= patient_bbox[1]
                and bbox[3] <= patient_bbox[3]
            ):
                inside_patient_bboxes.append(bbox)

        # NMS suppression
        filtered_bboxes = []
        for i, bbox1 in enumerate(inside_patient_bboxes):
            keep = True
            for j, bbox2 in enumerate(inside_patient_bboxes[:i]):
                # Only compare with higher-scoring boxes
                # Calculate IoU
                x1 = max(bbox1[0], bbox2[0])
                y1 = max(bbox1[1], bbox2[1])
                x2 = min(bbox1[2], bbox2[2])
                y2 = min(bbox1[3], bbox2[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                intersection = (x2 - x1) * (y2 - y1)
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                iou = intersection / (area1 + area2 - intersection)
                if iou > self.hand_iou_threshold:  # Significant overlap
                    keep = False
                    break
            if keep:
                filtered_bboxes.append(bbox1)
        return filtered_bboxes
    
    def _detect_patient_and_hands(self, image):
        """
        Locate the patient and hands in the provided image. Returns a dictionary with the following keys:
        - 'query_type': either 'person' or 'patient'. 'patient' is more trustworthy.
        - 'patient_bbox': [x1, y1, x2, y2] for patient bounding box
        - 'hand_bboxes': List of all detected hand bounding boxes, filtered to be inside
            the patient bounding box and non-maximum suppressed.

        Throws a ValueError if no patient or multiple patients are detected. No error
        is raised if no hands are detected, but the 'hand_bboxes' key may have an empty list.
        """
        result = {}

        for query_type in ["patient", "person"]:

            inputs = self.processor(
                images=image,
                text=f"{query_type}. hand.",  # Detect BOTH patient and hand!
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

            # Ensure only one patient is detected, appeal to authority if not
            if dino_results and len(dino_results[0]["text_labels"]) > 0:

                patient_indices = [
                    i
                    for i, label in enumerate(dino_results[0]["text_labels"])
                    if label == query_type
                ]

                if len(patient_indices) > 1:
                    raise PatientDetectionError(
                        f"Multiple patients detected with query type '{query_type}'"
                    )
                elif len(patient_indices) == 1:
                    result["query_type"] = query_type
                    # patient_score = dino_results[0]["scores"][patient_indices[0]].item()
                    patient_box = (
                        dino_results[0]["boxes"][patient_indices[0]]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )
                    result["patient_bbox"] = patient_box.tolist()
        
        if result.get("query_type") is None:
            raise PatientDetectionError(
                f"No patient detected with 'patient' or 'preson'. "
                f"Detected labels: {dino_results[0]['text_labels']}"
            )
        
        # Only reached when exactly one patient is detected
        hand_indices = [
            i for i, label in enumerate(dino_results[0]["text_labels"]) if label == "hand"
        ]
        hand_indices.sort(key=lambda i: dino_results[0]["scores"][i].item(), reverse=True)
        hand_boxes = [
            dino_results[0]["boxes"][i].cpu().numpy().astype(int).tolist() for i in hand_indices
        ]
        hand_boxes = self._filter_hand_bboxes(hand_boxes, result["patient_bbox"])
        result["hand_bboxes"] = hand_boxes
        return result
    
    def locate_patient_and_hands(self, image):
        """
        Find the left and right hands of a patient in an image using pose estimation.

        Args:
            image (numpy.ndarray): (H, W, 3) numpy array in RGB format.
        Returns:
            dict: A dictionary containing:
                - 'patient_keypoints': (17, 3) numpy array of COCO keypoints for the patient
                - 'left_hand': Bounding box for left hand, or None if not detected
                - 'right_hand': Bounding box for right hand, or None if not detected
                - Other keys from the _detect_patient_and_hands method
        Raises:
            MultiplePatientsDetectedError: If more than one patient is detected in the image
            NoPatientDetectedError: If no patients are detected in the image
        Notes:
            - ALL bounding boxes are in pixel space and [x1, y1, x2, y2] format as a list
            - Uses confidence thresholds to filter detections
            - When only one hand is detected, it's assigned based on which elbow is more visible
            - When multiple hands are detected, assignments are based on proximity to wrist keypoints
            - Prevents assigning the same box to both hands by using confidence scores as tiebreakers
        """

        result = self._detect_patient_and_hands(image)

        pose_results = self.pose_model(image, verbose=False)
        kpts = pose_results[0].keypoints.data.cpu().numpy()  # 0 index for one image
        if kpts.shape[0] > 1:
            raise PatientDetectionError("Multiple patients detected by pose model")
        if kpts.shape[0] == 0:
            raise PatientDetectionError("No patients detected by pose model")
        patient_kpts = kpts[0]
        result['patient_keypoints'] = patient_kpts

        # Verify that the pose is inside the patient bounding box
        num_inside = sum([
            distance_to_box(kpt[:2], result["patient_bbox"]) == 0 for kpt in patient_kpts
        ])
        if num_inside < self.coco_kpts_threshold:
            raise PatientDetectionError(
                f"Not enough COCO keypoints inside patient bounding box: {num_inside}"
            )

        left_elbow = patient_kpts[7]  # [x, y, confidence]
        right_elbow = patient_kpts[8]  # [x, y, confidence]
        left_wrist = patient_kpts[9]  # [x, y, confidence]
        right_wrist = patient_kpts[10]  # [x, y, confidence]

        # If there is only one hand box, check which arm is more visible
        hand_bboxes = result["hand_bboxes"]
        result["left_hand"] = None
        result["right_hand"] = None

        if len(hand_bboxes) == 0:
            return result

        if len(hand_bboxes) == 1:
            # Left elbow is index 7, right elbow is index 8
            left_elbow_confidence = left_elbow[2]
            right_elbow_confidence = right_elbow[2]
            if left_elbow_confidence > right_elbow_confidence:
                result["left_hand"] = hand_bboxes[0]
            else:
                result["right_hand"] = hand_bboxes[0]
            return result
        
        # If 2+ hand boxes, assign left/right wrist keypoints to nearest bbox
        # to get left/right hand boxes
        left_hand_box = None
        right_hand_box = None
        if (
            left_wrist[2] > self.confidence_threshold
        ):  # Only consider if confidence is reasonable
            distances = [distance_to_box(left_wrist, box) for box in hand_bboxes]
            if min(distances) < image.shape[0] / 4:  # Reasonable distance threshold
                left_hand_box = hand_bboxes[np.argmin(distances)]
        if (
            right_wrist[2] > self.confidence_threshold
        ):  # Only consider if confidence is reasonable
            distances = [distance_to_box(right_wrist, box) for box in hand_bboxes]
            if min(distances) < image.shape[0] / 4:  # Reasonable distance threshold
                right_hand_box = hand_bboxes[np.argmin(distances)]

        # Check if a box was assigned to both wrists
        if (
            left_hand_box is not None
            and right_hand_box is not None
            and np.array_equal(left_hand_box, right_hand_box)
        ):
            # If same box, assign to the wrist with higher confidence
            if left_wrist[2] > right_wrist[2]:
                right_hand_box = None
            else:
                left_hand_box = None

        result["left_hand"] = left_hand_box if left_hand_box is not None else None
        result["right_hand"] = right_hand_box if right_hand_box is not None else None
        return result


def annotate_hands(image, hand_result):
    """
    Draw bounding boxes, keypoints and labels for detected hands and patient

    Parameters:
        image (numpy.ndarray): The input image in RGB format
        hand_result (dict): Dictionary with hand boxes, patient box, and keypoints

    Returns:
        numpy.ndarray: Annotated image
    """
    import cv2

    # Convert image from RGB to BGR (for OpenCV display)
    image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    # Define colors (BGR format)
    left_color = (0, 0, 255)  # Red for left hand
    right_color = (255, 0, 0)  # Blue for right hand
    patient_color = (0, 255, 0)  # Green for patient
    hand_box_color = (255, 255, 0)  # Cyan for general hand boxes

    # Draw patient bounding box if available
    if "patient_bbox" in hand_result:
        x1, y1, x2, y2 = hand_result["patient_bbox"]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), patient_color, 2)
        text = "Patient"
        cv2.putText(
            image_bgr,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            patient_color,
            2,
        )

    # Draw all hand bounding boxes if available
    if "hand_bboxes" in hand_result:
        for i, box in enumerate(hand_result["hand_bboxes"]):
            x1, y1, x2, y2 = box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), hand_box_color, 1)
            cv2.putText(
                image_bgr,
                f"Hand {i}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                hand_box_color,
                1,
            )

    # Define keypoint indices and colors
    keypoints_to_draw = {
        # Left side (green)
        "left_shoulder": {"index": 5, "color": (0, 255, 0)},
        "left_elbow": {"index": 7, "color": (0, 255, 0)},
        "left_wrist": {"index": 9, "color": (0, 255, 0)},
        # Right side (red)
        "right_shoulder": {"index": 6, "color": (0, 0, 255)},
        "right_elbow": {"index": 8, "color": (0, 0, 255)},
        "right_wrist": {"index": 10, "color": (0, 0, 255)},
    }
    if "patient_keypoints" in hand_result:
        keypoint = hand_result["patient_keypoints"]
        for name, info in keypoints_to_draw.items():
            kpt = keypoint[info["index"]]

            # Draw keypoint if confidence is reasonable
            if kpt[2] > 0.2:  # Only show if confidence is above threshold
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(image_bgr, (x, y), 5, info["color"], -1)

                # Add text with confidence
                text = f"{name}: {kpt[2]:.2f}"
                cv2.putText(
                    image_bgr,
                    text,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    info["color"],
                    2,
                )

    # Draw left hand if detected
    if hand_result.get("left_hand"):
        x1, y1, x2, y2 = hand_result["left_hand"]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), left_color, 2)

        # Add label
        text = "Left Hand"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            image_bgr,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            left_color,
            -1,
        )
        cv2.putText(
            image_bgr,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # Draw right hand if detected
    if hand_result.get("right_hand"):
        x1, y1, x2, y2 = hand_result["right_hand"]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), right_color, 2)

        # Add label
        text = "Right Hand"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            image_bgr,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            right_color,
            -1,
        )
        cv2.putText(
            image_bgr,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return image_bgr
