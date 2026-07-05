import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import pickle

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("Loading preprocessed dataset...")
X_scaled = pd.read_csv("scaled_features.csv")
df_clustered = pd.read_csv("user_data_with_clusters.csv")

# Perform PCA to reduce behavior features to 2 dimensions
print("Running Principal Component Analysis (PCA)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Variance check
var_ratio = pca.explained_variance_ratio_
print(f"Explained Variance: PC1={var_ratio[0]*100:.2f}%, PC2={var_ratio[1]*100:.2f}% (Combined={sum(var_ratio)*100:.2f}%)")

# Save the trained PCA transformer
with open("models/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
print("Saved PCA transformer to 'models/pca_model.pkl'")

# Append PCA component scores to labeled dataset
df_clustered['pca_1'] = X_pca[:, 0]
df_clustered['pca_2'] = X_pca[:, 1]

df_clustered.to_csv("user_data_with_pca.csv", index=False)
print("Saved dataset with PCA components to 'user_data_with_pca.csv'")

# Visualize 2D PCA cluster space
print("Generating PCA scatter plot...")
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='pca_1', y='pca_2', 
    hue='cluster', 
    data=df_clustered, 
    palette='viridis', 
    alpha=0.6, 
    edgecolor=None,
    s=15
)

plt.title('User Segments in 2D PCA Space', fontsize=16, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)', fontsize=12)
plt.legend(title='Clusters', labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'], loc='best', frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("plots/pca_clusters.png", dpi=150)
plt.close()
print("Saved PCA cluster plot to 'plots/pca_clusters.png'")
print("PCA reduction complete.")
