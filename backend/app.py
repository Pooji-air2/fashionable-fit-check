from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("ml/fit_model.pkl")

# Load products
products = pd.read_csv("datasets/products.csv")


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

    # Selected product
    product_id = data["product_id"]

    product = products[products["product_id"] == product_id]

    if product.empty:
        return jsonify({
            "error": "Product not found"
        }), 404

    product = product.iloc[0]

    # Product category
    category = str(product["category"]).lower()

    # Product measurements
    product_chest = product["chest_cm"]
    product_waist = product["waist_cm"]
    product_hip = product["hip_cm"]
    product_shoulder = product["shoulder_cm"]

    # ------------------------------------------
    # Calculate differences based on clothing type
    # ------------------------------------------

    if category in ["jeans", "pants"]:

        # Bottom wear does not have shoulder/chest measurements
        chest_diff = 0
        shoulder_diff = 0

        waist_diff = product_waist - user_waist
        hip_diff = product_hip - user_hip

    else:

        # Tops use all body measurements
        chest_diff = product_chest - user_chest
        waist_diff = product_waist - user_waist
        hip_diff = product_hip - user_hip
        shoulder_diff = product_shoulder - user_shoulder

    # Overall difference
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
        "fit_result": str(prediction),
        "overall_difference": int(overall_difference),
        "product_id": product_id
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)