import joblib
import pandas as pd

# Load the trained ML model

model = joblib.load("ml/fit_model.pkl")
products = pd.read_csv("datasets/products.csv")

print("Fit prediction model loaded successfully!")


def predict_fit(chest_diff, waist_diff, hip_diff, shoulder_diff):

    overall_difference = (
        abs(chest_diff)
        + abs(waist_diff)
        + abs(hip_diff)
        + abs(shoulder_diff)
    )

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

    prediction = model.predict(features)[0]

    return prediction


# Test prediction
result = predict_fit(0, -2, 0, 1)

print("\nPredicted Fit:")
print(result)

def predict_fit(user_chest, user_waist, user_hip, user_shoulder,
                product_chest, product_waist, product_hip, product_shoulder):

    # Calculate measurement differences
    chest_diff = product_chest - user_chest
    waist_diff = product_waist - user_waist
    hip_diff = product_hip - user_hip
    shoulder_diff = product_shoulder - user_shoulder

    # Calculate overall difference
    overall_difference = (
        abs(chest_diff)
        + abs(waist_diff)
        + abs(hip_diff)
        + abs(shoulder_diff)
    )

    # Create features for the ML model
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

    return prediction


# Example user measurements
user = {
    "chest": 96,
    "waist": 92,
    "hip": 98,
    "shoulder": 44
}

# Example product measurements
product = {
    "chest": 96,
    "waist": 90,
    "hip": 98,
    "shoulder": 46
}

# Get prediction
result = predict_fit(
    user["chest"],
    user["waist"],
    user["hip"],
    user["shoulder"],
    product["chest"],
    product["waist"],
    product["hip"],
    product["shoulder"]
)

print("\nUser Measurements:")
print(user)

print("\nProduct Measurements:")
print(product)

print("\nPredicted Fit:")
print(result)

# Get user measurements
print("\nEnter your measurements:")

user_chest = float(input("Chest (cm): "))
user_waist = float(input("Waist (cm): "))
user_hip = float(input("Hip (cm): "))
user_shoulder = float(input("Shoulder (cm): "))


# Get product measurements
# Select a product
print("\nAvailable Products:")
print(products["product_id"].tolist())

product_id = input("\nEnter Product ID: ")

# Find the selected product
product = products[products["product_id"] == product_id]

if product.empty:
    print("Product not found!")
    exit()

product = product.iloc[0]

product_chest = product["chest_cm"]
product_waist = product["waist_cm"]
product_hip = product["hip_cm"]
product_shoulder = product["shoulder_cm"]

print("\nSelected Product:")
print(product_id)


# Predict fit
result = predict_fit(
    user_chest,
    user_waist,
    user_hip,
    user_shoulder,
    product_chest,
    product_waist,
    product_hip,
    product_shoulder
)

print("\n==============================")
print("     FASHIONABLE FIT CHECK")
print("==============================")

if result == "Perfect":
    print("👕 Recommended Fit: PERFECT")
    print("The product should fit you well.")

elif result == "Tight":
    print("👕 Recommended Fit: TIGHT")
    print("Consider choosing a larger size.")

elif result == "Loose":
    print("👕 Recommended Fit: LOOSE")
    print("Consider choosing a smaller size.")

print("==============================")