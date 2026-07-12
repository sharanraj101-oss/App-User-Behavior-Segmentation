import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

print("Loading raw dataset...")
df = pd.read_csv("user_behavior_dataset.csv")

# I clean duplicate entries
duplicates_count = df.duplicated().sum()
print(f"Duplicates found: {duplicates_count}")
if duplicates_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed successfully.")

# I impute missing user ratings with 0
print("Imputing missing ratings with 0...")
df['rating_given'] = df['rating_given'].fillna(0)

# I run a quick validation check
print(f"Total remaining missing values: {df.isnull().sum().sum()}")

# I export my cleaned dataset
df.to_csv("cleaned_user_data.csv", index=False)
print("Saved cleaned dataset to 'cleaned_user_data.csv'")

# I select behavioral features for my clustering
clustering_features = [
    'sessions_per_week', 
    'avg_session_duration_min', 
    'daily_active_minutes', 
    'feature_clicks_per_session', 
    'notifications_opened_per_week', 
    'pages_viewed_per_session', 
    'days_since_last_login', 
    'rating_given', 
    'churn_risk_score', 
    'engagement_score'
]

print(f"Features list: {clustering_features}")
X = df[clustering_features]

# I scale and standardize numerical features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# I convert it back to a DataFrame for verification and export
X_scaled_df = pd.DataFrame(X_scaled, columns=clustering_features)

print("\n--- Scaling Check ---")
for col in clustering_features:
    mean = X_scaled_df[col].mean()
    std = X_scaled_df[col].std()
    print(f"{col:<30} | Mean: {mean: .3f} | Std: {std: .3f}")

X_scaled_df.to_csv("scaled_features.csv", index=False)
print("\nSaved scaled features to 'scaled_features.csv'")
print("Preprocessing pipeline completed.")
