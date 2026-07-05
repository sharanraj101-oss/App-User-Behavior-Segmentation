import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import pickle

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load scaled features
print("Loading scaled features...")
X_scaled = pd.read_csv("scaled_features.csv")

# 1. Compute WCSS for Elbow Method
print("Running Elbow Method (k=1 to 10)...")
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    print(f"k={k} | Inertia: {kmeans.inertia_:.2f}")

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method to Determine Optimal k', fontsize=14)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/elbow_curve.png", dpi=150)
plt.close()
print("Saved Elbow Curve to 'plots/elbow_curve.png'")

# 2. Train final K-Means model with k=4 (as specified in PDF results)
print("\nTraining final K-Means model with k=4...")
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)

# Save the trained model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans_final, f)
print("Saved K-Means model to 'models/kmeans_model.pkl'")

# 3. Assign Cluster Labels to our Master Cleaned Dataset
print("Assigning cluster labels to cleaned user data...")
df_cleaned = pd.read_csv("cleaned_user_data.csv")
df_cleaned['cluster'] = kmeans_final.labels_

# Print cluster size distribution
print("\n--- Cluster Size Distribution ---")
cluster_counts = df_cleaned['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(df_cleaned)) * 100
    print(f"Cluster {cluster_id}: {count:<6} users ({percentage:.2f}%)")

# Save final clustered dataset
df_cleaned.to_csv("user_data_with_clusters.csv", index=False)
print("\nSaved clustered dataset to 'user_data_with_clusters.csv'")
print("Model training complete!")
