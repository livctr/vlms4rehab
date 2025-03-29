from flask import Flask, request, jsonify
from data.pipeline.backend.server.app_conf import (
    HUMAN_INPUT_JSON_PATH,
    dataset,
)

app = Flask(__name__)

import base64
import io
import json
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)


def create_dummy_image():
    # Create a dummy image (RGB) and a random binary mask (for demo purposes)
    H, W = 480, 640
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (H, W), dtype=np.uint8) * 255
    return image, mask

def image_to_base64(img: np.ndarray):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return encoded

# Endpoint to get the image and additional data
@app.route("/api/data", methods=["GET"])
def get_data():
    image, mask = create_dummy_image()
    img_base64 = image_to_base64(image)
    bounding_boxes = [
        {"id": 1, "x1": 100, "y1": 120, "x2": 150, "y2": 150},
        {"id": 2, "x1": 300, "y1": 200, "x2": 130, "y2": 130}
    ]
    prompt_text = "patient's left/right hand"
    return jsonify({
        "image": img_base64,
        "bounding_boxes": bounding_boxes,
        "prompt": prompt_text
    })

# Endpoint to receive the selected bounding box
@app.route("/api/submit", methods=["POST"])
def submit_selection():
    data = request.json
    selected_box = data.get("selected_box")
    if selected_box:
        print("Selected bounding box:", selected_box)
        # Here, process the selected bounding box as needed.
        return jsonify({"status": "success", "selected_box": selected_box})
    else:
        return jsonify({"status": "error", "message": "No bounding box selected"}), 400




if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the Flask app.")
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on')
    parser.add_argument('--ip', type=str, help='IP address to bind the Flask app to')
    args = parser.parse_args()
    # Run the app on all interfaces so it can be accessed remotely
    print(f"Running on {args.ip}:{args.port}")
    app.run(host=args.ip, port=args.port, debug=True)
