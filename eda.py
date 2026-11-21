
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualization folder
os.makedirs("Visualizations", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("Processed_Data/loan_train_cleaned.csv")

# Set chart style
sns.set_theme(style="whitegrid")
print("Dataset loaded:", df.shape)

# Check target distribution
print("\nLOAN STATUS DISTRIBUTION")
print(df["loan_status"].value_counts())

print("\nLOAN STATUS PERCENTAGE")
print((df["loan_status"].value_counts(normalize=True) * 100).round(2))

# Helper function to find available columns
def find_column(options):
    for col in options:
        if col in df.columns:
            return col
    return None

# Identify relevant columns
income_col = find_column(["income", "applicant_income"])
loan_col = find_column(["loan_amount", "loan_amnt"])
purpose_col = find_column(["loan_purpose", "purpose"])
employment_col = find_column([
    "employment_status", "employment",
    "emp_length", "self_employed"
])
credit_col = find_column([
    "credit_score", "credit_history", "credit_type"
])

# Function to save charts
def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        f"Visualizations/{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

# 1. Loan Status Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="loan_status")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
save_chart("01_Loan_Status_Distribution")

# 2. Applicant Income Distribution
if income_col:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=income_col,
        bins=40,
        color="steelblue"
    )
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Number of Applicants")
    save_chart("02_Applicant_Income")
else:
    print("Income column not found.")

# 3. Loan Amount Distribution
if loan_col:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=loan_col,
        bins=40,
        color="teal"
    )
    plt.title("Loan Amount Distribution")
    plt.xlabel("Loan Amount")
    plt.ylabel("Number of Applicants")
    save_chart("03_Loan_Amount")
else:
    print("Loan amount column not found.")

# 4. Employment Analysis
if employment_col:
    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=df,
        x=employment_col,
        hue="loan_status"
    )
    plt.title("Employment Status vs Loan Status")
    plt.xlabel("Employment Category")
    plt.ylabel("Number of Applicants")
    plt.xticks(rotation=30)
    save_chart("04_Employment_Analysis")
else:
    print("Employment column not found in this dataset.")

# 5. Credit Analysis
if credit_col:
    plt.figure(figsize=(9, 5))

    if pd.api.types.is_numeric_dtype(df[credit_col]):
        sns.histplot(
            data=df,
            x=credit_col,
            hue="loan_status",
            bins=30,
            multiple="stack"
        )
    else:
        sns.countplot(
            data=df,
            x=credit_col,
            hue="loan_status"
        )
        plt.xticks(rotation=30)

    plt.title("Credit Profile vs Loan Status")
    plt.xlabel("Credit Profile")
    plt.ylabel("Number of Applicants")
    save_chart("05_Credit_Analysis")
else:
    print("Credit column not found.")

# 6. Loan Purpose Analysis
if purpose_col:
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df,
        x=purpose_col,
        hue="loan_status"
    )
    plt.title("Loan Purpose vs Loan Status")
    plt.xlabel("Loan Purpose")
    plt.ylabel("Number of Applicants")
    plt.xticks(rotation=30)
    save_chart("06_Loan_Purpose_Analysis")
else:
    print("Loan purpose column not found.")

print("\nEDA COMPLETED!")
print("Charts saved in Visualizations folder.")
