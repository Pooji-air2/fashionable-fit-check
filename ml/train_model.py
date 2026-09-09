import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the expanded dataset
data = pd.read_csv("datasets/expanded_fit_data.csv")

print("\nML Dataset:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

# Separate features and target
X = data.drop("fit_result", axis=1)
y = data["fit_result"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")
¸
# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["Tight", "Perfect", "Loose"],
    zero_division=0
))

# Save model
joblib.dump(model, "ml/fit_model.pkl")

print("\nModel saved successfully!")