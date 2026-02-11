# --------------------------------------
# t-SNE : Non-linear Dimensionality Reduction
# --------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize the data
X_scaled = StandardScaler().fit_transform(X)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE result
plt.figure(figsize=(7,5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='cool')
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE - Dimensionality Reduction")
plt.colorbar()
plt.show()
