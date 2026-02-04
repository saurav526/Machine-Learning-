import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# -------------------------------
# 1. Generate synthetic data
# -------------------------------
X, y = make_blobs(
    n_samples=1000,
    centers=3,
    n_features=2,
    random_state=23
)

print("Shape of X:", X.shape)

# -------------------------------
# 2. Visualize generated data
# -------------------------------
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Data with 3 Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Elbow Method (WCSS)
# -------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.xticks(np.arange(1, 11, 1))
plt.show()

# -------------------------------
# 5. Knee (Elbow) Detection
# -------------------------------
kl = KneeLocator(
    range(1, 11),
    wcss,
    curve="convex",
    direction="decreasing"
)

print("Optimal number of clusters (Elbow):", kl.elbow)

# -------------------------------
# 6. KMeans with optimal clusters
# -------------------------------
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)

y_labels = kmeans.fit_predict(X_train)

# -------------------------------
# 7. Silhouette Score for k = 3
# -------------------------------
score = silhouette_score(X_train, y_labels)
print("Silhouette Score for k=3:", score)

# -------------------------------
# 8. Silhouette Scores for multiple k
# -------------------------------
for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    y_labels = kmeans.fit_predict(X_train)
    silhouette_avg = silhouette_score(X_train, y_labels)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg}")

# -------------------------------
# 9. Visualize KMeans Clusters
# -------------------------------
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_labels,
    cmap='viridis'
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='red',
    marker='X'
)
plt.title("KMeans Clustering Result (k=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
