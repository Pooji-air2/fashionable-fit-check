import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("datasets/products.csv")

samples = []

for _, product in data.iterrows():

    differences = [
        (-8, -8, -8, -2, "Loose"),
        (-5, -5, -5, -1, "Loose"),
        (-2, -2, -2, 0, "Perfect"),
        (0, 0, 0, 0, "Perfect"),
        (2, 2, 2, 1, "Perfect"),
        (5, 5, 5, 2, "Tight"),
        (8, 8, 8, 2, "Tight")
    ]

    for chest_diff, waist_diff, hip_diff, shoulder_diff, fit in differences:

        overall_difference = (
            abs(chest_diff)
            + abs(waist_diff)
            + abs(hip_diff)
            + abs(shoulder_diff)
        )

        samples.append([
            chest_diff,
            waist_diff,
            hip_diff,
            shoulder_diff,
            overall_difference,
            fit
        ])


# Create training dataframe
df = pd.DataFrame(samples, columns=[
    "chest_diff",
    "waist_diff",
    "hip_diff",
    "shoulder_diff",
    "overall_difference",
    "fit_type"
])


# Features
X = df[
    [
        "chest_diff",
        "waist_diff",
        "hip_diff",
        "shoulder_diff",
        "overall_difference"
    ]
]

# Target
y = df["fit_type"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(" Fashionable Fit Check - ML Model")
print("===================================")

print("\nDataset samples:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "ml/fit_model.pkl")

print("\nModel saved successfully!")
print("Location: ml/fit_model.pkl")