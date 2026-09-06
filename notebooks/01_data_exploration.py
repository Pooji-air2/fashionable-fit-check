
import pandas as pd

# Load the products dataset
products = pd.read_csv("datasets/products.csv")
# Display the dataset
print(products)

print("\nDataset Shape:")
print(products.shape)

print("\nColumn Names:")
print(products.columns)

print("\nDataset Information:")
print(products.info())

print("\nMissing Values:")
print(products.isnull().sum())

print("\nDuplicate Rows:")
print(products.duplicated().sum())

print("\nData Types:")
print(products.dtypes)

print("\nData Types:")
print(products.dtypes)

# Load the users dataset
users = pd.read_csv("datasets/users.csv")

print("\nUsers Dataset:")
print(users)

print("\nUsers Shape:")
print(users.shape)

print("\nUsers Missing Values:")
print(users.isnull().sum())

print("\nUsers Duplicate Rows:")
print(users.duplicated().sum())

# Load the fit dataset
fit_data = pd.read_csv("datasets/fit_data.csv")

print("\nFit Dataset:")
print(fit_data)

print("\nFit Dataset Shape:")
print(fit_data.shape)

print("\nFit Dataset Missing Values:")
print(fit_data.isnull().sum())

print("\nFit Dataset Duplicate Rows:")
print(fit_data.duplicated().sum())

print("\nFit Result Distribution:")
print(fit_data["fit_result"].value_counts())

print("\nFit Result Distribution:")
print(fit_data["fit_result"].value_counts())

print("\nProduct Measurement Statistics:")
print(products[[
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "length_cm",
    "sleeve_cm"
]].describe())

print("\nFit Analysis:")

fit_analysis = fit_data[[
    "user_id",
    "product_id",
    "fit_result"
]]

print(fit_analysis)

print("\nDay 2 Data Exploration Completed Successfully!")

# Fashionable Fit Check
# Day 2 - Data Exploration

import pandas as pd
import numpy as np

print("Fashionable Fit Check - Data Exploration")
print("Day 2 started successfully!")

