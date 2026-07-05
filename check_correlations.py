import pandas as pd

df = pd.read_csv("user_behavior_dataset.csv")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

corr = df[numerical_cols].corr()
print("=== Correlations with engagement_score ===")
print(corr['engagement_score'].sort_values(ascending=False))

print("\n=== Correlations with churn_risk_score ===")
print(corr['churn_risk_score'].sort_values(ascending=False))
