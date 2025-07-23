@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"result": "No file uploaded"}), 400

    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Count edge pixels in region where the "G" usually is
    height, width = edges.shape
    g_region = edges[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.7)]
    edge_count = np.sum(g_region > 0)

    # Threshold chosen heuristically; you can tune this
    if edge_count < 1000:
        result = "split"
    else:
        result = "not split"

    return jsonify({"result": result})
