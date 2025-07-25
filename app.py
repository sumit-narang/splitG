from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Split the G API is running"

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"result": "No file uploaded"}), 400

    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"result": "Failed to read image"}), 400

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    g_found = False
    g_split = False

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (40, 40))

        # Naive check: If the shape is roughly G-shaped (based on white pixel pattern)
        top = roi_resized[:20, :]
        bottom = roi_resized[20:, :]

        top_white = cv2.countNonZero(top)
        bottom_white = cv2.countNonZero(bottom)
        total_white = cv2.countNonZero(roi_resized)

        if total_white > 100:  # likely a letter
            g_found = True
            ratio = abs(top_white - bottom_white) / total_white
            if ratio < 0.2:
                g_split = True
                break

    if not g_found:
        return jsonify({"result": "No G detected"})

    if g_split:
        return jsonify({"result": "You split the G!"})
    else:
        return jsonify({"result": "You missed the G!"})
