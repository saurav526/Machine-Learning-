# --------------------------------------
# Isomap : Manifold Learning
# --------------------------------------

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Apply Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X_scaled)

# Plot result
plt.figure(figsize=(7,5))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap='cool')
plt.xlabel("Isomap Component 1")
plt.ylabel("Isomap Component 2")
plt.title("Isomap - Dimensionality Reduction")
plt.colorbar()
plt.show()
