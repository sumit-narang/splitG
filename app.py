from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin from React frontend

# Load template G image
template_path = os.path.join(os.path.dirname(__file__), 'template_g.png')
template_g = cv2.imread(template_path, 0)  # Grayscale

def is_g_split(cropped_g):
    _, thresh = cv2.threshold(cropped_g, 127, 255, cv2.THRESH_BINARY_INV)
    horizontal_proj = np.sum(thresh, axis=1)
    normalized = horizontal_proj / np.max(horizontal_proj)
    mid = len(normalized) // 2
    region = normalized[mid - mid//10 : mid + mid//10]
    if np.min(region) < 0.1:
        return True
    return False

@app.route('/check_g_split', methods=['POST'])
def check_g_split():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray, template_g, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val < threshold:
        return jsonify({'error': 'G not found in image'}), 404

    top_left = max_loc
    h, w = template_g.shape
    cropped_g = gray[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

    split = is_g_split(cropped_g)
    return jsonify({'g_split': split})

if __name__ == '__main__':
    app.run(debug=True)
