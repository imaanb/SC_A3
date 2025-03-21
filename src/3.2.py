import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

def laplacian_circle(n, sparse=True):
    """
    Create the Laplacian matrix for a circular grid with diameter n using finite difference method.
    :param n: Diameter of the circle (approximate number of points across the circle)
    :return: Sparse Laplacian matrix, grid points
    """
    if n == 1:
        matrix = sp.csr_matrix(([-4], ([0], [0])), shape=(1, 1)) if sparse else np.array([[-4]])
        return matrix, None 
    
    # Generate circular grid points
    radius = n / 2
    points = []
    for i in range(n):
        for j in range(n):
            x, y = i - radius, j - radius
            if x**2 + y**2 <= radius**2:
                points.append((i, j))
    
    num_points = len(points)
    L = np.zeros((num_points, num_points))
    
    # Build Laplacian using finite differences
    point_indices = {point: idx for idx, point in enumerate(points)}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-point stencil
    
    for (i, j), idx in point_indices.items():
        L[idx, idx] = -4
        for di, dj in directions:
            neighbor = (i + di, j + dj)
            if neighbor in point_indices:
                neighbor_idx = point_indices[neighbor]
                L[idx, neighbor_idx] = 1
    
    return (sp.csr_matrix(L), points) if sparse else (L, points)

# Solve the system
n = 100  # Grid size
M, points = laplacian_circle(n, sparse=True)

num_points = len(points)
b = np.zeros(num_points)

# Identify the source point (closest to (0.6, 1.2))
source_point = (round(n/2 + 0.6/2*n/2),round(n/2 + 1.2/2*n/2))
source_idx = points.index(source_point)
b[source_idx] = 1  # Set source condition

# Modify M to enforce Dirichlet condition at the source
M = M.tolil()
# M[source_idx, :] = 0
M[source_idx, source_idx] = 1
M = M.tocsr()

# Solve for concentration
c = spsolve(M, b)

# Convert to grid
C = np.zeros((n, n))
for (i, j), val in zip(points, c):
    C[i, j] = val

# Plot result
x_vals = np.linspace(-2, 2, n)
y_vals = np.linspace(-2, 2, n)
X, Y = np.meshgrid(x_vals, y_vals)

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, C, levels=20, cmap='viridis')
plt.colorbar(label='Concentration')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Steady-State Diffusion Solution')
plt.show()
