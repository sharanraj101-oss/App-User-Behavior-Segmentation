import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
print("Loading dataset...")
df = pd.read_csv("user_behavior_dataset.csv")

# 1. Check for duplicates
duplicates_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates_count}")
if duplicates_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# 2. Impute missing values for rating_given with 0 (Option B)
print("Imputing missing ratings with 0...")
df['rating_given'] = df['rating_given'].fillna(0)

# Verify no more missing values
print(f"Remaining missing values: {df.isnull().sum().sum()}")

# Save the cleaned master dataset (with all columns for profiling later)
df.to_csv("cleaned_user_data.csv", index=False)
print("Saved cleaned master dataset to 'cleaned_user_data.csv'")

# 3. Select numerical features for clustering (per PDF guidelines)
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

print(f"Selected {len(clustering_features)} features for clustering: {clustering_features}")
X = df[clustering_features]

# 4. Scale features using StandardScaler
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to DataFrame for inspection and saving
X_scaled_df = pd.DataFrame(X_scaled, columns=clustering_features)

# Print scaling verification (Mean should be ~0, Std should be ~1)
print("\n--- Scaling Verification (Mean & Std Dev) ---")
for col in clustering_features:
    mean = X_scaled_df[col].mean()
    std = X_scaled_df[col].std()
    print(f"{col:<30} | Mean: {mean: .3f} | Std Dev: {std: .3f}")

# Save scaled features
X_scaled_df.to_csv("scaled_features.csv", index=False)
print("\nSaved scaled features to 'scaled_features.csv'")
print("Preprocessing complete!")
