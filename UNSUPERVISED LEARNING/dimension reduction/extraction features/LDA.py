# --------------------------------------
# LDA : Feature Extraction (Supervised)
# --------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize the data
X_scaled = StandardScaler().fit_transform(X)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Plot LDA result
plt.figure(figsize=(7,5))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='plasma')
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.title("LDA - Feature Extraction")
plt.colorbar()
plt.show()
