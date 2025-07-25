from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

def detect_split_by_template(image):
    # Load template
    template_path = "g_template.png"
    if not os.path.exists(template_path):
        return False, "Template not found"

    template = cv2.imread(template_path, 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Template matching
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.5:
        return None, "Template match too weak"

    top_left = max_loc
    h, w = template.shape
    g_roi = image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

    # Analyze horizontal line through center
    center_y = h // 2
    horizontal_slice = g_roi[center_y-2:center_y+2, :]

    gray_slice = cv2.cvtColor(horizontal_slice, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray_slice, 200, 255, cv2.THRESH_BINARY)

    white_pixels = np.sum(binarized == 255)
    total_pixels = binarized.size
    white_pixel_ratio = white_pixels / total_pixels

    print("Template white pixel ratio:", white_pixel_ratio)
    return white_pixel_ratio > 0.2, "Template method used"

def fallback_split_detection(image):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Count white pixels along each row
    row_sums = np.sum(binarized == 255, axis=1)
    max_row = np.argmax(row_sums)

    h, w = binarized.shape
    center_y = h // 2

    print("Foam line Y:", max_row, "Center Y:", center_y)
    difference = abs(max_row - center_y)

    # Acceptable error margin (tweak as needed)
    return difference <= 10, "Fallback method used"

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"result": "No file provided"}), 400

    file = request.files["file"]
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    result, reason = detect_split_by_template(image)
    if result is None:
        # Fallback if template method fails
        result, reason = fallback_split_detection(image)

    final_result = "You split the G!" if result else "Try again!"
    return jsonify({"result": final_result, "method": reason})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
