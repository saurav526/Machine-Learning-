from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, :2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

centers = model.cluster_centers_

print("Cluster Labels:")
print(clusters)

print("Inertia:", model.inertia_)

plt.scatter(
    X_scaled[:, 0], X_scaled[:, 1],
    c=clusters, cmap="viridis"
)

plt.scatter(
    centers[:, 0], centers[:, 1],
    color="red", marker="X", s=200,
    label="Centroids"
)

plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")
plt.title("K-Means Clustering - Iris Dataset")
plt.legend()
plt.show()
