from flask import Flask, request, jsonify
import numpy as np
import os
import traceback

from src.utils import load_object

app = Flask(__name__)

model = None


def load_model():
    global model
    model_path = os.path.join(os.getcwd(), "artifacts", "model", "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run training pipeline first.")
    model = load_object(model_path)


@app.route("/", methods=["GET"])
def home():
    return "🚀 Customer Categorizer Model Deployment is running. Use POST /predict with JSON payload."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Support both list and dict formats
        if isinstance(data, dict):
            features = np.array(list(data.values())).reshape(1, -1)
        elif isinstance(data, list):
            features = np.array(data).reshape(1, -1)
        else:
            return jsonify({"error": "Invalid input format, send list or dict"}), 400

        preds = model.predict(features)
        return jsonify({"prediction": preds.tolist()}), 200
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e), "trace": traceback_str}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
