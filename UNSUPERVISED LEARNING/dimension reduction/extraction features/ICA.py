# --------------------------------------
# ICA : Independent Component Analysis
# --------------------------------------

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Apply ICA
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# Plot ICA result
plt.figure(figsize=(7,5))
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=y, cmap='viridis')
plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("ICA - Dimensionality Reduction")
plt.colorbar()
plt.show()
