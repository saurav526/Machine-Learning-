import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from spicy.cluster.hierarchy import dendrogram, linkage

X, y = make_blobs(
    n_samples=1000,
    centers=3,      
    n_features=2,   
    random_state=23
)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Data with 3 Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

linked = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Sample Index")      
plt.ylabel("Cluster Distance")    
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

agg_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_pred = agg_cluster.fit_predict(X_train)

sil_score = silhouette_score(X_train, y_pred)
print("Silhouette Score for Agglomerative Clustering:", sil_score)


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred) 
plt.title("Agglomerative Clustering Results")
plt.xlabel("Feature 1") 
plt.ylabel("Feature 2")
plt.legend()
plt.show()