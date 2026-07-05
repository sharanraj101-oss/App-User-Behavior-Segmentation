import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import pickle

# Create directory for plots and models if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load data
print("Loading data...")
X_scaled = pd.read_csv("scaled_features.csv")
df_clustered = pd.read_csv("user_data_with_clusters.csv")

# 1. Run PCA with 2 components
print("Running Principal Component Analysis (PCA)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 2. Print variance ratios
var_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: PC1={var_ratio[0]*100:.2f}%, PC2={var_ratio[1]*100:.2f}% (Total={sum(var_ratio)*100:.2f}%)")

# Save PCA model
with open("models/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
print("Saved PCA model to 'models/pca_model.pkl'")

# 3. Add PCA components to the clustered dataframe
df_clustered['pca_1'] = X_pca[:, 0]
df_clustered['pca_2'] = X_pca[:, 1]

# Save dataset with PCA coordinates
df_clustered.to_csv("user_data_with_pca.csv", index=False)
print("Saved dataset with PCA components to 'user_data_with_pca.csv'")

# 4. Visualize the 4 clusters in the PCA space
print("Generating PCA scatter plot...")
plt.figure(figsize=(12, 8))
# Use a distinct, premium color palette for our 4 clusters
sns.scatterplot(
    x='pca_1', y='pca_2', 
    hue='cluster', 
    data=df_clustered, 
    palette='viridis', 
    alpha=0.6, 
    edgecolor=None,
    s=15
)

plt.title('Visualization of User Segments in PCA Space', fontsize=16, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)', fontsize=12)
plt.legend(title='Clusters', labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'], loc='best', frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot
plt.tight_layout()
plt.savefig("plots/pca_clusters.png", dpi=150)
plt.close()
print("Saved PCA cluster plot to 'plots/pca_clusters.png'")
print("PCA analysis complete!")
