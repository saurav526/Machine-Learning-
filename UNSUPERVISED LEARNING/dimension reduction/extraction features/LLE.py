# --------------------------------------
# LLE : Locally Linear Embedding
# --------------------------------------

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Apply LLE
lle = LocallyLinearEmbedding(
    n_neighbors=10,
    n_components=2,
    method='standard'
)
X_lle = lle.fit_transform(X_scaled)

# Plot result
plt.figure(figsize=(7,5))
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y, cmap='spring')
plt.xlabel("LLE Component 1")
plt.ylabel("LLE Component 2")
plt.title("LLE - Dimensionality Reduction")
plt.colorbar()
plt.show()
