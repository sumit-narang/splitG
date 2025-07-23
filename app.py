from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React frontend

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Read image with OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Dummy logic to simulate "split the G" detection
    height, width = img.shape[:2]
    center_pixel = img[height // 2, width // 2]
    is_split = center_pixel[1] > 100  # Example heuristic: check green value

    result = "You split the G!" if is_split else "Not quite split."

    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Fix for Railway's $PORT
    app.run(host="0.0.0.0", port=port)
