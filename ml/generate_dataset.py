import pandas as pd
import random

random.seed(42)

data = []

# Generate 100 Perfect fits
for i in range(100):

    user_chest = random.randint(84, 112)
    user_waist = random.randint(70, 100)
    user_hip = random.randint(88, 116)
    user_shoulder = random.randint(36, 48)

    product_chest = user_chest + random.randint(-2, 2)
    product_waist = user_waist + random.randint(-2, 2)
    product_hip = user_hip + random.randint(-2, 2)
    product_shoulder = user_shoulder + random.randint(-1, 1)

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

    data.append([
        chest_diff,
        waist_diff,
        hip_diff,
        shoulder_diff,
        overall_difference,
        "Perfect"
    ])


# Generate 100 Loose fits
for i in range(100):

    user_chest = random.randint(84, 112)
    user_waist = random.randint(70, 100)
    user_hip = random.randint(88, 116)
    user_shoulder = random.randint(36, 48)

    product_chest = user_chest + random.randint(6, 12)
    product_waist = user_waist + random.randint(6, 12)
    product_hip = user_hip + random.randint(6, 12)
    product_shoulder = user_shoulder + random.randint(1, 3)

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

    data.append([
        chest_diff,
        waist_diff,
        hip_diff,
        shoulder_diff,
        overall_difference,
        "Loose"
    ])


# Generate 100 Tight fits
for i in range(100):

    user_chest = random.randint(84, 112)
    user_waist = random.randint(70, 100)
    user_hip = random.randint(88, 116)
    user_shoulder = random.randint(36, 48)

    product_chest = user_chest - random.randint(6, 12)
    product_waist = user_waist - random.randint(6, 12)
    product_hip = user_hip - random.randint(6, 12)
    product_shoulder = user_shoulder - random.randint(1, 3)

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

    data.append([
        chest_diff,
        waist_diff,
        hip_diff,
        shoulder_diff,
        overall_difference,
        "Tight"
    ])


# Create DataFrame
df = pd.DataFrame(data, columns=[
    "chest_diff",
    "waist_diff",
    "hip_diff",
    "shoulder_diff",
    "overall_difference",
    "fit_result"
])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset
df.to_csv("datasets/expanded_fit_data.csv", index=False)

print("Balanced dataset created successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFit Result Distribution:")
print(df["fit_result"].value_counts())