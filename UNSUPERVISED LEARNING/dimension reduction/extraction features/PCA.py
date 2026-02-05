import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_iris()
X = data.data
y = data.target

X_scaled = StandardScaler().fit_transform(X)

# apply PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0],X_pca[:,1], c=y)
plt.xlabel("principle components 1")
plt.ylabel("principle components 2")
plt.title("PCA of Iris Dataset")
plt.legend(data.target_names)  
plt.show()