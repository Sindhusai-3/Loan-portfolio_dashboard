
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# Create folders
os.makedirs("ML_Models", exist_ok=True)
os.makedirs("Processed_Data", exist_ok=True)

# 1. Load prepared datasets
X_train = pd.read_csv(
    "Processed_Data/X_train_prepared.csv"
)

X_val = pd.read_csv(
    "Processed_Data/X_val_prepared.csv"
)

y_train = pd.read_csv(
    "Processed_Data/y_train.csv"
).squeeze("columns").astype(int)

y_val = pd.read_csv(
    "Processed_Data/y_val.csv"
).squeeze("columns").astype(int)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

# 2. Define four ML models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        min_samples_split=20,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# 3. Train and evaluate models
results = []
validation_predictions = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print("TRAINING:", name)
    print("=" * 50)

    # Balance minority class for Gradient Boosting
    if name == "Gradient Boosting":
        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )
    else:
        model.fit(X_train, y_train)

    # Predict validation classes
    y_pred = model.predict(X_val)

    # Predict probability of class 1
    class_one_index = list(model.classes_).index(1)

    y_probability = model.predict_proba(X_val)[
        :, class_one_index
    ]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val, y_pred, pos_label=1, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred, pos_label=1, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred, pos_label=1, zero_division=0
    )

    roc_auc = roc_auc_score(y_val, y_probability)

    pr_auc = average_precision_score(
        y_val, y_probability
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    validation_predictions[name] = y_probability

    # Save each trained model
    filename = (
        name.lower().replace(" ", "_") + ".pkl"
    )

    joblib.dump(
        model,
        "ML_Models/" + filename
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_val, y_pred, zero_division=0
    ))

# 4. Compare all models
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))

# 5. Save comparison results
comparison.to_csv(
    "Processed_Data/model_comparison.csv",
    index=False
)

# 6. Save validation probabilities
probability_df = pd.DataFrame(
    validation_predictions
)

probability_df["Actual_Status"] = y_val.values

probability_df.to_csv(
    "Processed_Data/validation_predictions.csv",
    index=False
)
best_model_name = comparison.iloc[0]["Model"]
print("\nBEST MODEL BASED ON ROC-AUC:")
print(best_model_name)
with open("ML_Models/best_model.txt", "w") as file:
    file.write(best_model_name)

print("\nALL MODELS TRAINED SUCCESSFULLY!")
print("All models saved in ML_Models folder.")
print("Comparison saved in Processed_Data folder.")
