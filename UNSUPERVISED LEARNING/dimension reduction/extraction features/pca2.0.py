import numpy as np

# Step 1: Create dataset (example)
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Step 2: Mean centering (subtract mean)
mean = np.mean(X, axis=0)
X_centered = X - mean

# Step 3: Covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Step 4: Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort eigenvalues and eigenvectors
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Step 6: Select top k eigenvectors (k=1 or 2)
k = 1
principal_components = sorted_eigenvectors[:, :k]

# Step 7: Linear transformation (project data)
X_reduced = np.dot(X_centered, principal_components)



# Output results
print("Mean:\n", mean)
print("\nCovariance Matrix:\n", cov_matrix)
print("\nEigenvalues:\n", sorted_eigenvalues)
print("\nEigenvectors:\n", sorted_eigenvectors)
print("\nReduced Data:\n", X_reduced)