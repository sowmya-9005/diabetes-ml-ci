from pathlib import Path
import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("diabetes_model.pkl")

# sklearn diabetes dataset features
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "diabetes_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "diabetes-progression-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in FEATURES if f not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = np.array([[data[f] for f in FEATURES]])
    model = load_model()
    prediction = float(model.predict(sample)[0])

    return jsonify({
        "prediction": round(prediction, 2),
        "description": "Predicted diabetes progression score"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
