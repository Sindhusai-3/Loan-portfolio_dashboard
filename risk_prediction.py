
import pandas as pd
import numpy as np
import joblib
import os

# Create output folder
os.makedirs("Processed_Data", exist_ok=True)

# 1. Load best model
model = joblib.load(
    "ML_Models/gradient_boosting.pkl"
)

# 2. Load prepared test dataset
X_test = pd.read_csv(
    "Processed_Data/X_test_prepared.csv"
)

# 3. Load original test IDs
test_ids = pd.read_csv(
    "Processed_Data/test_ids.csv"
)

print("Test Records:", len(X_test))

# 4. Predict loan status
predicted_status = model.predict(X_test)

# 5. Calculate probability of class 1
class_one_index = list(model.classes_).index(1)

default_probability = model.predict_proba(
    X_test
)[:, class_one_index]

# 6. Assign risk categories
risk_category = np.select(
    [
        default_probability < 0.30,
        default_probability < 0.60
    ],
    [
        "Low Risk",
        "Medium Risk"
    ],
    default="High Risk"
)

# 7. Create prediction output
results = pd.DataFrame({
    "id": test_ids["id"],
    "Predicted_Status": predicted_status,
    "Default_Probability": default_probability,
    "Default_Probability_Percentage":
        default_probability * 100,
    "Risk_Category": risk_category
})

# 8. Save predictions
results.to_csv(
    "Processed_Data/loan_risk_predictions.csv",
    index=False
)

# 9. Display summary
print("\nRISK PREDICTION SUMMARY")

print("\nRisk Categories:")
print(results["Risk_Category"].value_counts())

print("\nAverage Predicted Class-1 Probability:")
print(round(default_probability.mean() * 100, 2), "%")

print("\nFirst 10 Predictions:")
print(results.head(10))

print("\nSTEP 7 COMPLETED SUCCESSFULLY!")

print(
    "\nSaved: Processed_Data/loan_risk_predictions.csv"
)
