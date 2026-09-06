from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained ML model
model = joblib.load("ml/fit_model.pkl")


@app.route("/")
def home():
    return "Fashionable Fit Check API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # User measurements
    user_chest = data["user_chest"]
    user_waist = data["user_waist"]
    user_hip = data["user_hip"]
    user_shoulder = data["user_shoulder"]

    # Product measurements
    product_chest = data["product_chest"]
    product_waist = data["product_waist"]
    product_hip = data["product_hip"]
    product_shoulder = data["product_shoulder"]

    # Calculate differences
    chest_diff = product_chest - user_chest
    waist_diff = product_waist - user_waist
    hip_diff = product_hip - user_hip
    shoulder_diff = product_shoulder - user_shoulder

    overall_difference = (
        abs(chest_diff)
        + abs(waist_diff)
        + abs(hip_diff)
        + abs(shoulder_diff)
    )

    # Create ML features
    features = pd.DataFrame([[
        chest_diff,
        waist_diff,
        hip_diff,
        shoulder_diff,
        overall_difference
    ]], columns=[
        "chest_diff",
        "waist_diff",
        "hip_diff",
        "shoulder_diff",
        "overall_difference"
    ])

    # Predict fit
    prediction = model.predict(features)[0]

    return jsonify({
        "fit_result": prediction,
        "overall_difference": overall_difference
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)