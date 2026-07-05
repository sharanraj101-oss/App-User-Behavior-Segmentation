import pandas as pd

df = pd.read_csv("user_behavior_dataset.csv")

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

for col in behavioral_cols:
    print(f"--- {col} ---")
    print(df[col].describe())
    print()
