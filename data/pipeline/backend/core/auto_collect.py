from data.pipeline.backend.core.hand_predictor import HandPredictor
from data.pipeline.backend.core.data_manager import DataManager
from data.pipeline.backend.core.collection_api import AutoAPI

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the automatic annotator")
    parser.add_argument('--dino_model_id', type=str, default="IDEA-Research/grounding-dino-base", 
                        help='Model ID for DINO')
    parser.add_argument('--pose_model_id', type=str, default="yolo11x-pose.pt", 
                        help='Model ID for pose detection')
    parser.add_argument('--hand_iou_threshold', type=float, default=0.5, 
                        help='IoU threshold for hand detection')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                        help='Confidence threshold for detection')
    parser.add_argument('--coco_kpts_threshold', type=int, default=3, 
                        help='Keypoints threshold for COCO')
    parser.add_argument('--device', type=str, default=None, 
                        help='Computation device (e.g., "cuda", "cpu")')
    parser.add_argument('--annotation_frequency', type=int, default=5, 
                        help='Annotation frequency in seconds')
    parser.add_argument('--sampling_fps', type=int, default=8, 
                        help='Sampling frames per second')

    args = parser.parse_args()
    
    hand_predictor = HandPredictor(
        dino_model_id=args.dino_model_id,
        pose_model_id=args.pose_model_id,
        hand_iou_threshold=args.hand_iou_threshold,
        confidence_threshold=args.confidence_threshold,
        coco_kpts_threshold=args.coco_kpts_threshold,
        device=args.device
    )
    human_input_data_manager = DataManager(
        annotation_frequency_s=args.annotation_frequency,
        sampling_fps=args.sampling_fps
    )

    auto_api = AutoAPI(
        hand_predictor=hand_predictor,
        data_manager=human_input_data_manager
    )
    auto_api.run()
