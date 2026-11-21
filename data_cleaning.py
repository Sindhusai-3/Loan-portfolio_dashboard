
import pandas as pd
import numpy as np
import os
os.makedirs("Processed_Data", exist_ok=True)
train = pd.read_csv("Dataset/loan_train.csv")
test = pd.read_csv("Dataset/loan_test.csv")

print("Original Training Shape:", train.shape)
print("Original Testing Shape:", test.shape)
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()
train = train.drop_duplicates()
test = test.drop_duplicates()
train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)
for df in [train, test]:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        df[col] = df[col].replace("", np.nan)
train = train.dropna(subset=["loan_status"])
target = "loan_status"
feature_cols = [col for col in train.columns if col != target]
missing_cols = set(feature_cols) - set(test.columns)

if missing_cols:
    raise ValueError(f"Missing columns in test dataset: {missing_cols}")

test = test[feature_cols]
numeric_cols = train[feature_cols].select_dtypes(
    include=np.number
).columns

for col in numeric_cols:
    median_value = train[col].median()

    if pd.isna(median_value):
        median_value = 0

    train[col] = train[col].fillna(median_value)
    test[col] = test[col].fillna(median_value)

# 8. Handle missing categorical values
categorical_cols = [
    col for col in feature_cols if col not in numeric_cols
]

for col in categorical_cols:
    mode_values = train[col].mode(dropna=True)

    fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"

    train[col] = train[col].fillna(fill_value)
    test[col] = test[col].fillna(fill_value)

# 9. Save cleaned datasets
train.to_csv(
    "Processed_Data/loan_train_cleaned.csv",
    index=False
)

test.to_csv(
    "Processed_Data/loan_test_cleaned.csv",
    index=False
)

# 10. Final report
print("\nDATA CLEANING COMPLETED!")

print("\nCleaned Training Shape:", train.shape)
print("Cleaned Testing Shape:", test.shape)

print("\nRemaining Missing Values in Training:",
      train.isnull().sum().sum())

print("Remaining Missing Values in Testing:",
      test.isnull().sum().sum())

print("\nTarget Distribution:")
print(train["loan_status"].value_counts())

print("\nFiles saved successfully!")
