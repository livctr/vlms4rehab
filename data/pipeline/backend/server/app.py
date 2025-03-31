from flask import Flask, request, jsonify
from data.pipeline.backend.core.hand_predictor import HandPredictor
from data.pipeline.backend.core.human_input_data_manager import HumanInputDataManager
from data.pipeline.backend.core.collection_api import CollectionAPI

import base64
import io
import json
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

hand_predictor = None
human_input_data_manager = None
collection_api = None


# Endpoint to get the image and additional data
@app.route("/api/data", methods=["GET"])
def get_data():
    # Get navigation parameter (previous, current, or next)
    navigation = request.args.get('navigation', 'current')
    print(f"Navigation: {navigation}")
    
    if not collection_api:
        return jsonify({"error": "Collection API not initialized"}), 500
    
    try:
        # Get data from collection API
        data = collection_api.get_data(navigation)
        return jsonify(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Endpoint to select a hand bounding box
@app.route("/api/select_hand_bbox", methods=["POST"])
def select_hand_bbox():
    if not collection_api:
        return jsonify({"error": "Collection API not initialized"}), 500
    
    try:
        human_input = request.json
        collection_api.select_hand_bbox(human_input)
        return collection_api.get_data("next")
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Endpoint to save annotations
@app.route("/api/save", methods=["POST"])
def save_annotations():
    if not collection_api:
        return jsonify({"error": "Collection API not initialized"}), 500
    
    try:
        result = collection_api.save()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the Flask app.")
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on')
    parser.add_argument('--ip', type=str, help='IP address to bind the Flask app to')
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
    parser.add_argument('--annotation_frequency', type=int, default=10, 
                        help='Annotation frequency in seconds')
    parser.add_argument('--sampling_fps', type=int, default=8, 
                        help='Sampling frames per second')

    args = parser.parse_args()
    
    # Initialize with command line arguments
    hand_predictor = HandPredictor(
        dino_model_id=args.dino_model_id,
        pose_model_id=args.pose_model_id,
        hand_iou_threshold=args.hand_iou_threshold,
        confidence_threshold=args.confidence_threshold,
        coco_kpts_threshold=args.coco_kpts_threshold,
        device=args.device
    )
    human_input_data_manager = HumanInputDataManager(
        annotation_frequency_s=args.annotation_frequency,
        sampling_fps=args.sampling_fps
    )
    collection_api = CollectionAPI(
        hand_predictor=hand_predictor,
        data_manager=human_input_data_manager
    )

    print(f"Running on {args.ip}:{args.port}")
    app.run(host=args.ip, port=args.port, debug=False)
