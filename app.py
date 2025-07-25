import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"result": "No image uploaded"}), 400

    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    probable_g = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if 50 < area < 1000 and 0.5 < aspect_ratio < 1.2:
            probable_g = (x, y, w, h)
            break

    if not probable_g:
        return jsonify({"result": "G not found"}), 200

    x, y, w, h = probable_g
    roi = gray[y:y+h, x:x+w]

    edges = cv2.Canny(roi, 100, 200)
    mid_y = h // 2
    band = edges[mid_y-2:mid_y+2, :]
    ratio = np.sum(band > 0) / band.size

    if ratio > 0.15:
        return jsonify({"result": "split"})
    else:
        return jsonify({"result": "not split"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
