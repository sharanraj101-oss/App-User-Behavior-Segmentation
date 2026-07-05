import pandas as pd

# Load the dataset
df = pd.read_csv("user_behavior_dataset.csv")

print("--- Dataset Shape ---")
print(f"Number of rows (users): {df.shape[0]}")
print(f"Number of columns (features): {df.shape[1]}")
print()

print("--- Column Information and Data Types ---")
print(df.dtypes)
print()

print("--- First 5 Rows ---")
print(df.head())
print()

print("--- Missing Values Count ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")
print()

print("--- Descriptive Statistics for Numerical Features ---")
print(df.describe())
print()

print("--- Unique Values in Categorical Features ---")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].unique()[:10]} (Total unique: {df[col].nunique()})")
