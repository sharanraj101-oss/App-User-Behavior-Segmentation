import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import pickle

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("Loading scaled features dataset...")
X_scaled = pd.read_csv("scaled_features.csv")

# Determine optimal clusters via Elbow Method
print("Running Elbow Method (k=1 to 10)...")
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    print(f"k={k} | Inertia: {kmeans.inertia_:.2f}")

# Plot elbow curve results
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Curve Analysis', fontsize=14)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/elbow_curve.png", dpi=150)
plt.close()
print("Saved Elbow Curve plot to 'plots/elbow_curve.png'")

# Train final K-Means model using k=4
print("\nFitting final K-Means model (k=4)...")
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)

# Serialize the trained model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans_final, f)
print("Saved trained K-Means model to 'models/kmeans_model.pkl'")

# Label the dataset with assigned clusters
print("Assigning cluster labels to cleaned user data...")
df_cleaned = pd.read_csv("cleaned_user_data.csv")
df_cleaned['cluster'] = kmeans_final.labels_

print("\n--- Cluster Size Distribution ---")
cluster_counts = df_cleaned['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(df_cleaned)) * 100
    print(f"Cluster {cluster_id}: {count:<6} users ({percentage:.2f}%)")

df_cleaned.to_csv("user_data_with_clusters.csv", index=False)
print("\nSaved output dataset to 'user_data_with_clusters.csv'")
print("Model training pipeline complete.")
