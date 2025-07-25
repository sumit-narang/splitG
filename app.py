# app.py
import cv2
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

def detect_split_g(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Template matching fallback
    template_path = 'g_template.png'
    if os.path.exists(template_path):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.6:
            # Extract matched region
            h, w = template.shape
            matched_region = thresh[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]

            # Check if horizontal split in matched region
            mid_row = h // 2
            top_half = matched_region[:mid_row, :]
            bottom_half = matched_region[mid_row:, :]

            # Count white pixels (inverted binary, so white = foreground)
            top_white = cv2.countNonZero(top_half)
            bottom_white = cv2.countNonZero(bottom_half)

            # If the top half and bottom half are clearly separated (low pixel overlap)
            if top_white > 10 and bottom_white > 10:
                # Check if there's a "gap" horizontally around the middle row
                gap_row = matched_region[mid_row - 2:mid_row + 2, :]
                gap_white = cv2.countNonZero(gap_row)
                if gap_white < (w * 0.1):
                    return True

    # Fallback: contour analysis to find horizontal split line
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 40:
            roi = thresh[y:y+h, x:x+w]
            # Check horizontal white "line" inside roi
            horizontal_sum = np.sum(roi, axis=1) / 255  # count white pixels per row
            min_val = np.min(horizontal_sum)
            min_idx = np.argmin(horizontal_sum)
            if min_val < w * 0.1 and 10 < min_idx < (h - 10):
                # A strong horizontal gap inside contour -> likely split G
                return True

    return False

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"result": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"result": "No file selected"}), 400

    # Read image from memory
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"result": "Invalid image"}), 400

    try:
        split_detected = detect_split_g(image)
        if split_detected:
            return jsonify({"result": "split detected"})
        else:
            return jsonify({"result": "no split detected"})
    except Exception as e:
        return jsonify({"result": f"error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
