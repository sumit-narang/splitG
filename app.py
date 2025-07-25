from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
import os

app = Flask(__name__)
CORS(app)

# Load the G template (if exists)
TEMPLATE_PATH = "g_template.png"
g_template = cv2.imread(TEMPLATE_PATH, 0) if os.path.exists(TEMPLATE_PATH) else None

def find_g_with_template(image_gray):
    if g_template is None:
        return None

    res = cv2.matchTemplate(image_gray, g_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.6:
        return None  # Not confident

    h, w = g_template.shape
    top_left = max_loc
    return image_gray[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

def check_split(roi):
    h, w = roi.shape
    mid_y = h // 2
    horizontal_band = roi[mid_y-5:mid_y+5, :]
    _, binary = cv2.threshold(horizontal_band, 200, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(binary == 255) / binary.size
    return white_ratio > 0.2

def fallback_ocr_check(image):
    text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(text_data['text'])

    for i in range(n_boxes):
        if text_data['text'][i].strip().upper() == 'G':
            (x, y, w, h) = (text_data['left'][i], text_data['top'][i],
                            text_data['width'][i], text_data['height'][i])
            roi = image[y:y+h, x:x+w]
            if check_split(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)):
                return True
    return False

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = "Try Again!"
    method = ""

    g_roi = find_g_with_template(gray)
    if g_roi is not None and check_split(g_roi):
        result = "Split the G"
        method = "Template"
    elif fallback_ocr_check(image):
        result = "Split the G"
        method = "OCR fallback"
    else:
        method = "OCR fallback or Template failed"

    return jsonify({"result": result, "method": method})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
