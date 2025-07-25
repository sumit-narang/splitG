from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the template image of G (must exist)
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'template_g.png')
template_g = cv2.imread(TEMPLATE_PATH, 0)  # Grayscale

if template_g is None:
    raise FileNotFoundError("template_g.png not found or unreadable")

def is_g_split(cropped_g):
    # Convert to binary
    _, thresh = cv2.threshold(cropped_g, 127, 255, cv2.THRESH_BINARY_INV)

    # Get horizontal projection
    horizontal_proj = np.sum(thresh, axis=1)
    normalized = horizontal_proj / np.max(horizontal_proj)

    mid = len(normalized) // 2
    region = normalized[mid - mid // 10 : mid + mid // 10]

    # Detect dip in the middle (split)
    return np.min(region) < 0.1

@app.route('/check_g_split', methods=['POST'])
def check_g_split():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Match template
    res = cv2.matchTemplate(gray, template_g, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.6:
        return jsonify({'error': 'G not found'}), 404

    h, w = template_g.shape
    top_left = max_loc
    cropped_g = gray[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

    split = is_g_split(cropped_g)
    return jsonify({'g_split': split})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
