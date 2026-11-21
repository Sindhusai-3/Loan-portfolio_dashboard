
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
os.makedirs("Processed_Data", exist_ok=True)
os.makedirs("ML_Models", exist_ok=True)
train = pd.read_csv("Dataset/loan_train.csv")
test = pd.read_csv("Dataset/loan_test.csv")

print("Training Dataset:", train.shape)
print("Testing Dataset:", test.shape)
target = "loan_status"

y = train[target]
X = train.drop(columns=[target])
numeric_features = [
    "loan_amnt",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "pub_rec_bankruptcies"
]
categorical_features = [
    "term",
    "grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose"
]
numeric_features = [
    c for c in numeric_features
    if c in X.columns and c in test.columns
]

categorical_features = [
    c for c in categorical_features
    if c in X.columns and c in test.columns
]

features = numeric_features + categorical_features

X = X[features].copy()
X_test = test[features].copy()
for df in [X, X_test]:
    for col in ["revol_util"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("%", "", regex=False)
            )
for df in [X, X_test]:
    for col in numeric_features:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )
y = pd.to_numeric(y, errors="coerce")
valid = y.notna()

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].astype(int).reset_index(drop=True)
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape)
print("Validation Features:", X_val.shape)
print("Testing Features:", X_test.shape)
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)
X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation and test data
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# 12. Get feature names
feature_names = preprocessor.get_feature_names_out()

# Convert into DataFrames
X_train_df = pd.DataFrame(
    X_train_processed, columns=feature_names
)

X_val_df = pd.DataFrame(
    X_val_processed, columns=feature_names
)

X_test_df = pd.DataFrame(
    X_test_processed, columns=feature_names
)

# 13. Save prepared datasets
X_train_df.to_csv(
    "Processed_Data/X_train_prepared.csv", index=False
)

X_val_df.to_csv(
    "Processed_Data/X_val_prepared.csv", index=False
)

X_test_df.to_csv(
    "Processed_Data/X_test_prepared.csv", index=False
)

y_train.to_csv(
    "Processed_Data/y_train.csv", index=False
)

y_val.to_csv(
    "Processed_Data/y_val.csv", index=False
)
test[["id"]].to_csv(
    "Processed_Data/test_ids.csv", index=False
)
joblib.dump(
    preprocessor,
    "ML_Models/preprocessor.pkl"
)

print("\nFEATURE PREPARATION COMPLETED!")

print("Number of numerical features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))
print("Total encoded features:", len(feature_names))

print("\nTarget distribution:")
print(y.value_counts())

print("\nSaved files successfully!")
