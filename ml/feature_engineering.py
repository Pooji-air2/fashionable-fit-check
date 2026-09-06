import pandas as pd

# Load datasets
users = pd.read_csv("datasets/users.csv")
products = pd.read_csv("datasets/products.csv")
fit_data = pd.read_csv("datasets/fit_data.csv")

# Combine user and product information
data = fit_data.merge(users, on="user_id")
data = data.merge(products, on="product_id")

print("\nCombined Dataset:")
print(data)

print("\nCombined Dataset Shape:")
print(data.shape)

# Create measurement difference features

data["chest_diff"] = data["chest_cm_x"] - data["chest_cm_y"]
data["waist_diff"] = data["waist_cm_x"] - data["waist_cm_y"]
data["hip_diff"] = data["hip_cm_x"] - data["hip_cm_y"]
data["shoulder_diff"] = data["shoulder_cm_x"] - data["shoulder_cm_y"]

print("\nMeasurement Differences:")
print(data[[
    "user_id",
    "product_id",
    "chest_diff",
    "waist_diff",
    "hip_diff",
    "shoulder_diff",
    "fit_result"
]])

# Calculate overall measurement difference

data["overall_difference"] = (
    abs(data["chest_diff"]) +
    abs(data["waist_diff"]) +
    abs(data["hip_diff"]) +
    abs(data["shoulder_diff"])
)

print("\nOverall Fit Difference:")
print(data[[
    "user_id",
    "product_id",
    "chest_diff",
    "waist_diff",
    "hip_diff",
    "shoulder_diff",
    "overall_difference",
    "fit_result"
]])

# Select features for ML

features = data[[
    "chest_diff",
    "waist_diff",
    "hip_diff",
    "shoulder_diff",
    "overall_difference"
]]

target = data["fit_result"]

print("\nML Features:")
print(features)

print("\nTarget:")
print(target)

# Convert fit labels into numerical values

target_encoded = target.map({
    "Tight": 0,
    "Perfect": 1,
    "Loose": 2
})

print("\nEncoded Target:")
print(target_encoded)

# Create final ML dataset
ml_data = features.copy()
ml_data["fit_result"] = target_encoded

# Save the engineered dataset
ml_data.to_csv("datasets/ml_features.csv", index=False)

print("\nML feature dataset saved successfully!")
print(ml_data)