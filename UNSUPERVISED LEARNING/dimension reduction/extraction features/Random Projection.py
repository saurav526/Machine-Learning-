# --------------------------------------
# Random Projection
# --------------------------------------

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Apply Random Projection
rp = GaussianRandomProjection(n_components=2, random_state=42)
X_rp = rp.fit_transform(X_scaled)

# Plot result
plt.figure(figsize=(7,5))
plt.scatter(X_rp[:, 0], X_rp[:, 1], c=y, cmap='plasma')
plt.xlabel("Random Projection 1")
plt.ylabel("Random Projection 2")
plt.title("Random Projection - Dimensionality Reduction")
plt.colorbar()
plt.show()
