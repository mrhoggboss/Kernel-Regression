import numpy as np

# Define the eigenvalues
eigenvalues = np.array([0, 0.1, 0.995])

# Construct the diagonal matrix Lambda
Lambda = np.diag(eigenvalues)

# Define the given orthogonal matrix Q
Q = np.array([
    [-np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3],
    [np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3],
    [0, np.sqrt(6)/3, np.sqrt(3)/3]
])

# Ensure Q is orthogonal
assert np.allclose(Q.T @ Q, np.eye(3)), "Q is not orthogonal"

# Form the initial Laplacian L
L = Q @ Lambda @ Q.T

# Adjust L to have non-positive off-diagonal elements
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        if i != j:
            L[i, j] = -abs(L[i, j])

# Adjust the diagonal elements to ensure row sum is zero
for i in range(L.shape[0]):
    L[i, i] = -np.sum(L[i, :]) + L[i, i]

# Print the resulting Laplacian matrix
print("Graph Laplacian L:")
print(L)

# Verify the eigenvalues
eigvals = np.linalg.eigvals(L)
print("Eigenvalues of L:")
print(np.sort(eigvals))