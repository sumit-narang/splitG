from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

app = Flask(__name__)

def analyze_image(image_bytes):
    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)

    # Convert RGB to BGR (OpenCV uses BGR)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # OCR config to get text with boxes
    data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)

    # Find the letter G in the text with bounding boxes
    # We'll check if G is split horizontally (meaning gap inside letter G)

    # Extract all bounding boxes of letter 'G' or 'g'
    g_boxes = []
    for i, text in enumerate(data['text']):
        if text.strip().lower() == 'g':
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            g_boxes.append((x, y, w, h))

    if not g_boxes:
        return "Could not find letter G in the image."

    # For each G found, analyze its image part for horizontal split (gap)
    for (x, y, w, h) in g_boxes:
        roi = img_cv[y:y+h, x:x+w]

        # Convert ROI to grayscale and threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Sum pixels horizontally to detect horizontal gaps
        horizontal_sum = np.sum(thresh, axis=1) / 255  # counts black pixels per row

        # Look for rows with very low black pixels (gap rows)
        gap_rows = np.where(horizontal_sum < (w * 0.1))[0]  # less than 10% black pixels

        if len(gap_rows) > 3:  # arbitrary threshold of gap lines
            return "You Split the G!"

    return "Try Again!"

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({"result": "No file uploaded"}), 400

    file = request.files['file']
    img_bytes = file.read()

    result = analyze_image(img_bytes)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
