"""
Digit Classifier Backend
Run: pip install flask flask-cors scikit-learn numpy
Then: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE, static_url_path="")
CORS(app)

# ── Load model artifacts ──────────────────────────────────────────────────────
with open(os.path.join(BASE, "RandomForestClassifier.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE, "features.pkl"), "rb") as f:
    features = pickle.load(f)

print(f"✅  Model loaded — {model.n_estimators} trees, classes {model.classes_.tolist()}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
        { "pixels": [p0, p1, ..., p783] }   <- 784 float values 0-255
    Returns:
        { "prediction": 7, "probabilities": [0.01, 0.02, ...], "confidence": 0.87 }
    """
    data = request.get_json(force=True)

    if "pixels" not in data:
        return jsonify({"error": "Missing 'pixels' field"}), 400

    pixels = np.array(data["pixels"], dtype=float).reshape(1, 784)

    if pixels.shape[1] != 784:
        return jsonify({"error": f"Expected 784 pixels, got {pixels.shape[1]}"}), 400

    pixels_scaled = scaler.transform(pixels)
    prediction    = int(model.predict(pixels_scaled)[0])
    probabilities = model.predict_proba(pixels_scaled)[0].tolist()

    return jsonify({
        "prediction":    prediction,
        "probabilities": probabilities,
        "confidence":    max(probabilities)
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier", "classes": 10})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  Server running → http://localhost:5000\n")
    app.run(debug=True, port=5000)
