
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

# Create folder
os.makedirs("Visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Load model comparison results
comparison = pd.read_csv(
    "Processed_Data/model_comparison.csv"
)

print("\nMODEL COMPARISON RESULTS")
print(comparison.to_string(index=False))

# 2. Plot performance metrics
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC"
]

for metric in metrics:

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=comparison,
        x="Model",
        y=metric
    )

    plt.title(metric + " Comparison")
    plt.xlabel("Machine Learning Models")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.ylim(0, 1)
    plt.tight_layout()

    filename = (
        "Visualizations/07_"
        + metric.lower().replace("-", "_").replace(" ", "_")
        + "_comparison.png"
    )

    plt.savefig(filename, dpi=300)
    plt.close()

# 3. Load validation data
X_val = pd.read_csv(
    "Processed_Data/X_val_prepared.csv"
)

y_val = pd.read_csv(
    "Processed_Data/y_val.csv"
).squeeze("columns").astype(int)

# 4. Define saved model paths
model_files = {
    "Logistic Regression":
        "ML_Models/logistic_regression.pkl",

    "Decision Tree":
        "ML_Models/decision_tree.pkl",

    "Random Forest":
        "ML_Models/random_forest.pkl",

    "Gradient Boosting":
        "ML_Models/gradient_boosting.pkl"
}

# 5. Create confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (name, path) in zip(axes.flatten(), model_files.items()):

    model = joblib.load(path)

    predictions = model.predict(X_val)

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.suptitle("Confusion Matrix Comparison")
plt.tight_layout()

plt.savefig(
    "Visualizations/13_Confusion_Matrix_Comparison.png",
    dpi=300
)

plt.close()

# 6. Create ROC curves
plt.figure(figsize=(9, 7))

for name, path in model_files.items():

    model = joblib.load(path)

    probabilities = model.predict_proba(X_val)

    class_one_index = list(model.classes_).index(1)

    y_probability = probabilities[:, class_one_index]

    fpr, tpr, thresholds = roc_curve(
        y_val,
        y_probability,
        pos_label=1
    )

    roc_score = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_score:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    "Visualizations/14_ROC_Curve_Comparison.png",
    dpi=300
)

plt.close()

# 7. Print classification reports
print("\nCLASSIFICATION REPORTS")

for name, path in model_files.items():

    model = joblib.load(path)

    predictions = model.predict(X_val)

    print("\n", "=" * 45)
    print(name)
    print("=" * 45)

    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0
        )
    )

# 8. Identify best model by ROC-AUC
best_model = comparison.sort_values(
    "ROC-AUC",
    ascending=False
).iloc[0]

print("\nBEST MODEL BASED ON ROC-AUC")
print(best_model["Model"])

print("\nROC-AUC:", round(best_model["ROC-AUC"], 4))

# 9. Save final comparison
comparison.sort_values(
    "ROC-AUC",
    ascending=False
).to_csv(
    "Processed_Data/final_model_comparison.csv",
    index=False
)
print("\nSTEP 6 COMPLETED SUCCESSFULLY!")
print("All comparison charts saved in Visualizations folder.")
