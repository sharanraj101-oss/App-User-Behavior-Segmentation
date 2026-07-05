import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load data
df = pd.read_csv("user_behavior_dataset.csv")

# Set seaborn style for clean, beautiful plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Generating plots...")

# 1. Distribution of Engagement and Churn Risk Scores
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['engagement_score'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Engagement Score', fontsize=14)
axes[0].set_xlabel('Engagement Score')

sns.histplot(df['churn_risk_score'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribution of Churn Risk Score', fontsize=14)
axes[1].set_xlabel('Churn Risk Score')
plt.tight_layout()
plt.savefig("plots/engagement_churn_distributions.png", dpi=150)
plt.close()
print("Saved plots/engagement_churn_distributions.png")

# 2. Key Activity Metrics (Sessions & Duration)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['sessions_per_week'], kde=True, ax=axes[0], color='teal')
axes[0].set_title('Sessions Per Week', fontsize=14)

sns.histplot(df['avg_session_duration_min'], kde=True, ax=axes[1], color='purple')
axes[1].set_title('Avg Session Duration (min)', fontsize=14)
plt.tight_layout()
plt.savefig("plots/behavior_distributions.png", dpi=150)
plt.close()
print("Saved plots/behavior_distributions.png")

# 3. Correlation Matrix of Numerical Features
behavioral_cols = [
    'sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes', 
    'feature_clicks_per_session', 'notifications_opened_per_week', 
    'pages_viewed_per_session', 'days_since_last_login', 'rating_given', 
    'churn_risk_score', 'engagement_score', 'account_age_days'
]
corr = df[behavioral_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of User Behavior Features", fontsize=16)
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", dpi=150)
plt.close()
print("Saved plots/correlation_matrix.png")

# 4. Subscription Type and Device Distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(x='subscription_type', data=df, ax=axes[0], hue='subscription_type', legend=False, palette='Set2')
axes[0].set_title('Users by Subscription Type', fontsize=14)

sns.countplot(x='device_type', data=df, ax=axes[1], hue='device_type', legend=False, palette='Set3')
axes[1].set_title('Users by Device Type', fontsize=14)
plt.tight_layout()
plt.savefig("plots/categorical_distributions.png", dpi=150)
plt.close()
print("Saved plots/categorical_distributions.png")

# 5. Rating Given distribution (to see missing value context)
plt.figure(figsize=(8, 5))
# Since rating_given has missing values, dropna() is implicitly done by countplot
sns.countplot(x='rating_given', data=df, hue='rating_given', legend=False, palette='viridis')
plt.title('Distribution of User Ratings (Excluding Missing)', fontsize=14)
plt.xlabel('Rating Given')
plt.tight_layout()
plt.savefig("plots/ratings_distribution.png", dpi=150)
plt.close()
print("Saved plots/ratings_distribution.png")

print("All plots generated successfully!")
