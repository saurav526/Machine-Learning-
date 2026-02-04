import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split    

X,y = make_blobs(
    n_samples=1000, 
    centers=3, 
    n_features=2,
    cluster_std=0.6,
    random_state=23
    
)

plt.scatter(X[:,0], X[:,1], c=y)
plt.title ("Generated Data with 3 Clusters") 
plt.xlabel("Feature 1")
plt.ylabel("features")
plt.show()

DBSCAN = DBSCAN(eps=0.5, min_samples=5)
y_pred = DBSCAN.fit_predict(X)  

n_cluster = len(set(y_db)) - (1 if -1 in y_db else 0)
print("Number of clusters found by DBSCAN:", n_cluster)


if n_clusters > 1:
    score = silhouette_score(X[y_db != -1], y_db[y_db != -1])
    print("Silhouette Score (DBSCAN):", score)
else:
    print("Silhouette Score not defined")

plt.scatter(X[:,0], X[:,1], c=y_pred)
plt.title("DBSCAN Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()




