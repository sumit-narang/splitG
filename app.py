from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pytesseract
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_stream = io.BytesIO(file.read())
    image = Image.open(image_stream).convert("RGB")
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    open_cv_image = cv2.resize(open_cv_image, (600, int(open_cv_image.shape[0] * 600 / open_cv_image.shape[1])))

    foam_line_y = detect_foam_line(open_cv_image)
    g_center_y = detect_g_center_y(open_cv_image)

    if g_center_y is None or foam_line_y is None:
        result = "Could not detect the G or foam line"
    elif abs(foam_line_y - g_center_y) < 20:
        result = "you split the G"
    else:
        result = "you missed it"

    return jsonify({"result": result})

def detect_foam_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return None

    horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 10]
    if not horizontal_lines:
        return None

    y_coords = [line[0][1] for line in horizontal_lines]
    return int(np.median(y_coords))

def detect_g_center_y(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    for i, word in enumerate(data["text"]):
        if word.lower() == "g":
            top = data["top"][i]
            height = data["height"][i]
            return top + height // 2

    return None

if __name__ == "__main__":
    app.run(debug=True)
