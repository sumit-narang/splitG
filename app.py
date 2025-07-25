from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app)

def is_g_split(image):
    """
    Try to detect if the foam line horizontally splits the 'G' in Guinness.
    This is a simple approximation using template matching and edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Load a template of "G" (you should provide this image in your backend folder)
    template_path = os.path.join(os.path.dirname(__file__), "g_template.png")
    if not os.path.exists(template_path):
        return False  # Cannot detect without template

    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

    # 2. Template matching to find the "G" in the image
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where(res >= threshold)

    if len(loc[0]) == 0:
        return False  # No G detected

    # Take first matched "G" area
    pt = (loc[1][0], loc[0][0])
    g_roi = gray[pt[1]:pt[1]+h, pt[0]:pt[0]+w]

    # 3. Check if there's a bright horizontal line across the middle of G
    middle_y = g_roi.shape[0] // 2
    horizontal_slice = g_roi[middle_y-2:middle_y+2, :]  # narrow band across center

    # Count white-ish pixels
    _, binarized = cv2.threshold(horizontal_slice, 180, 255, cv2.THRESH_BINARY)
    white_pixel_ratio = np.sum(binarized == 255) / binarized.size

    return white_pixel_ratio > 0.3  # adjustable threshold

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"result": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"result": "Empty filename"}), 400

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        file.save(temp.name)
        image = cv2.imread(temp.name)
        os.unlink(temp.name)

    if image is None:
        return jsonify({"result": "Failed to read image"}), 500

    # Run the analysis
    split = is_g_split(image)

    return jsonify({
        "result": "You split the G!" if split else "Try again!"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
