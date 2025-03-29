from ultralytics import YOLO
import os
import numpy as np

# Ground dino
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os

from loguru import logger


def run_dino(image, processor, dino_model, device, text_prompt = "patient. hand."):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    dino_results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.shape[:-1]],
    )
    if not dino_results or len(dino_results[0]["text_labels"]) == 0:
        raise ValueError("No patient, person, or hand detected by Grounding DINO")
    return dino_results


def locate_patient_hands(
    image,
    processor,
    dino_model,
    pose_model,
    num_coco_kpts_threshold: int = 3,
    hand_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
):
    """
    Locates the left and right hands of a patient in an image.

    Args:
        image: RGB image as numpy array, shape (H, W, 3)
        processor: AutoProcessor for Grounding DINO
        dino_model: AutoModelForZeroShotObjectDetection model
        pose_model: YOLO pose estimation model
        num_coco_kpts_threshold: Minimum number of COCO keypoints inside patient bbox
            to be considered the patient's pose. If multiple poses are detected, raises
            an error.
        hand_iou_threshold: Threshold for IoU when performing NMS on hand boxes (default: 0.5)
        confidence_threshold: Minimum confidence for keypoints to be considered valid (default: 0.2)

    Returns:
        A dictionary containing:
        - 'query_type': either 'person' or 'patient'. 'patient' is more trustworthy.
        - 'patient_bbox': [x1, y1, x2, y2] for patient bounding box
        - 'patient_keypoints': COCO keypoints for the patient 
        - 'hand_bboxes': List of all detected hand bounding boxes
        - 'left_hand': [x1, y1, x2, y2] for left hand or None if not detected
        - 'right_hand': [x1, y1, x2, y2] for right hand or None if not detected
        At least one of 'left_hand' or 'right_hand' is not None if an error is not raised.
        Coordinates are *un-normalized* and in pixel space.

    Raises:
        ValueError: If the patient, their pose, or hand bounding boxes cannot be determined.

    Known Limitations:
        Detecting pose from birds-eye view. E.g., S0003_brushing4_2
        Can detect the wrong person as the patient.
        - Occurs when the highest confidence box is not the patient.
        
        Gets the wrong patient: E.g., S00023_combing5_1, C0005_combing5_1

    """
    result = {}

    device = dino_model.device

    # Step 1: Find patient and hand bounding boxes using Grounding DINO
    dino_results = run_dino(image, processor, dino_model, device, text_prompt="patient. hand.")
    result['query_type'] = "patient"
    # Extract patient bounding box (highest confidence patient)
    patient_indices = [
        i
        for i, label in enumerate(dino_results[0]["text_labels"])
        if label == "patient"
    ]
    if not patient_indices:
        # If no patient, try "person" as a fallback.
        dino_results = run_dino(image, processor, dino_model, device, text_prompt="person. hand.")
        patient_indices = [
            i for i, label in enumerate(dino_results[0]["text_labels"]) if label == "person"
        ]
        result['query_type'] = "person"
    if not patient_indices:
        raise ValueError("No patient detected by Grounding DINO")

    patient_scores = [dino_results[0]["scores"][i].item() for i in patient_indices]
    highest_patient_idx = patient_indices[torch.tensor(patient_scores).argmax().item()]
    patient_box = (
        dino_results[0]["boxes"][highest_patient_idx].cpu().numpy().astype(int)
    )
    result["patient_bbox"] = patient_box.tolist()

    # Step 2: Get pose keypoints
    pose_results = pose_model(image, verbose=False)

    if len(pose_results) == 0 or len(pose_results[0].keypoints.data) == 0:
        raise ValueError("No poses detected by YOLO")

    # Find pose corresponding to patient bbox
    all_keypoints = pose_results[0].keypoints.data.cpu().numpy()

    # Count keypoints inside patient bounding box for each detected person
    patient_pose_idx = -1

    for pose_idx, keypoints in enumerate(all_keypoints):

        # Count keypoints inside patient bounding box
        inside_count = sum(
            (
                point[2] > confidence_threshold
            )  # Filter to only include keypoints with high confidence
            and (point[0] >= patient_box[0])
            and (point[0] <= patient_box[2])
            and (point[1] >= patient_box[1])
            and (point[1] <= patient_box[3])
            for point in keypoints
        )
        print(f"Pose {pose_idx} has {inside_count} keypoints inside patient bbox")

        if inside_count > num_coco_kpts_threshold:
            if patient_pose_idx != -1:
                raise ValueError(
                    "Multiple poses detected within patient bounding box"
                )
            patient_pose_idx = pose_idx

    if patient_pose_idx == -1:
        raise ValueError("No pose found inside patient bounding box")

    patient_keypoints = all_keypoints[patient_pose_idx]
    result["patient_keypoints"] = patient_keypoints
    left_elbow = patient_keypoints[7]  # [x, y, confidence]
    right_elbow = patient_keypoints[8]  # [x, y, confidence]
    left_wrist = patient_keypoints[9]  # [x, y, confidence]
    right_wrist = patient_keypoints[10]  # [x, y, confidence]

    # Step 3: Extract hand bounding boxes
    hand_indices = [
        i for i, label in enumerate(dino_results[0]["text_labels"]) if label == "hand"
    ]

    if not hand_indices:
        raise ValueError("No hand bounding boxes detected")

    # Sort hand boxes by score (highest first)
    hand_indices.sort(key=lambda i: dino_results[0]["scores"][i].item(), reverse=True)
    hand_boxes = [
        dino_results[0]["boxes"][i].cpu().numpy().astype(int) for i in hand_indices
    ]
    result["hand_bboxes"] = [box.tolist() for box in hand_boxes]

    # Perform non-maximum suppression on hand boxes
    # Remove boxes outside the patient bounding box
    kept_boxes = []
    for i, box1 in enumerate(hand_boxes):

        # Check if box is completely outside of patient box
        patient_box = result["patient_bbox"]
        if (box1[2] < patient_box[0] or  # box is completely to the left
            box1[0] > patient_box[2] or  # box is completely to the right
            box1[3] < patient_box[1] or  # box is completely above
            box1[1] > patient_box[3]):   # box is completely below
            continue

        keep = True
        for j, box2 in enumerate(
            hand_boxes[:i]
        ):
            
            # Only compare with higher-scoring boxes
            # Calculate IoU
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x2 <= x1 or y2 <= y1:  # No overlap
                continue

            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            iou = intersection / (area1 + area2 - intersection)

            if iou > hand_iou_threshold:  # Significant overlap
                keep = False
                break

        if keep:
            kept_boxes.append(box1)

    if not kept_boxes:
        raise ValueError(
            "No hand bboxs remained after non-maximum suppression and filtering"
        )

    # If there's only one hand box, check which arm is more visible
    if len(kept_boxes) == 1:
        # Left elbow is index 7, right elbow is index 8
        left_elbow_confidence = left_elbow[2]
        right_elbow_confidence = right_elbow[2]

        if left_elbow_confidence > right_elbow_confidence:
            result["left_hand"] = kept_boxes[0].tolist()
            result["right_hand"] = None
        else:
            result["right_hand"] = kept_boxes[0].tolist()
            result["left_hand"] = None

        return result

    # If multiple hand boxes, assign to nearest wrist keypoint

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

    # For each wrist, find closest box if wrist confidence is high enough
    left_hand_box = None
    right_hand_box = None

    if (
        left_wrist[2] > confidence_threshold
    ):  # Only consider if confidence is reasonable
        distances = [distance_to_box(left_wrist, box) for box in kept_boxes]
        if min(distances) < image.shape[0] / 4:  # Reasonable distance threshold
            left_hand_box = kept_boxes[np.argmin(distances)]

    if (
        right_wrist[2] > confidence_threshold
    ):  # Only consider if confidence is reasonable
        distances = [distance_to_box(right_wrist, box) for box in kept_boxes]
        if min(distances) < image.shape[0] / 4:  # Reasonable distance threshold
            right_hand_box = kept_boxes[np.argmin(distances)]

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

    result["left_hand"] = left_hand_box.tolist() if left_hand_box is not None else None
    result["right_hand"] = right_hand_box.tolist() if right_hand_box is not None else None

    if left_hand_box is None and right_hand_box is None:
        raise ValueError(
            "Could not associate any hand bounding boxes with wrist keypoints"
        )

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


if __name__ == "__main__":
    # Set up models and processors
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    processor = AutoProcessor.from_pretrained(model_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device
    )
    pose_model = YOLO("yolo11x-pose.pt")

    # Create output directory if it doesn't exist
    output_dir = "hand_detection_result_images"
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files from the test_images folder
    test_images_dir = "test_images"
    test_images = [
        f
        for f in os.listdir(test_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
    ]
    test_images = ["S0003_brushing4_2.jpg"]
    print(f"Found {len(test_images)} images to process")

    # Process each image
    for img_file in test_images:
        print(f"Processing {img_file}...")
        try:
            # Load and convert image
            img_path = os.path.join(test_images_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load {img_file}, skipping")
                continue

            # Convert from BGR to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect hands
            hand_result = locate_patient_hands(
                image_rgb, processor, dino_model, pose_model
            )
            print(f"Detection results: {hand_result}")

            # Annotate image with hand detections
            annotated_image = annotate_hands(image_rgb, hand_result)

            # Save the result
            output_path = os.path.join(output_dir, f"hands_{img_file}")
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved result to {output_path}")

        except ValueError as e:
            print(f"Error processing {img_file}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error processing {img_file}: {str(e)}")
