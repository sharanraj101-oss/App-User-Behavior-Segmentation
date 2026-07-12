import pandas as pd
import numpy as np

# I load my clustered dataset
df = pd.read_csv("user_data_with_pca.csv")

# I set pandas display options to print all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# My list of behavioral columns
behavioral_cols = [
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

print("=== AVERAGE BEHAVIORAL METRICS PER CLUSTER ===")
profile = df.groupby('cluster')[behavioral_cols].mean()
print(profile)

print("\n=== DEMOGRAPHICS (AGE) PER CLUSTER ===")
age_profile = df.groupby('cluster')[['age', 'account_age_days']].mean()
print(age_profile)
