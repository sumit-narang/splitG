from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image
import numpy as np
import cv2
import io

# Set Tesseract path (for Docker)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (600, int(img_cv.shape[0] * 600 / img_cv.shape[1])))

        # Dummy logic for now
        text = pytesseract.image_to_string(img_cv).lower()
        g_found = "g" in text

        return jsonify({
            "result": "you split the G" if g_found else "you missed it"
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def health():
    return "Backend is up!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
