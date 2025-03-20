
from scipy.sparse import diags
from scipy.sparse.linalg import eigs, eigsh
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def laplacian_grid_square(n, viz = False):
    """
    Create the Laplacian matrix for an n x n grid using finite difference method.
    :param n: Grid size (n x n)
    :return: Sparse Laplacian matrix of shape (n^2, n^2)
    """
    
    size = n * n
    diagonals = []
    
    # Main diagonal (-4)
    diagonals.append(-4 * np.ones(size))
    
    # Right and left neighbor diagonals (+1)
    diagonals.append(np.ones(size - 1))
    diagonals.append(np.ones(size - 1))
    
    # Top and bottom neighbor diagonals (+1)
    diagonals.append(np.ones(size - n))
    diagonals.append(np.ones(size - n))
    
    # Convert to sparse matrix
    offsets = [0, 1, -1, n, -n]
    L = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    
    # Fix row edges (remove horizontal connections between rows)
    for i in range(1, n):
        L[i * n, i * n - 1] = 0
        L[i * n - 1, i * n] = 0
    
    if viz:
        # Visualize the Laplacian matrix as a heatmap
        plt.imshow(L.toarray(), cmap='viridis', interpolation='none')
        plt.colorbar(label='Matrix Value')
        plt.title(f"Laplacian Matrix Heatmap for {n}x{n} Grid")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.show()
    
    return L

def laplacian_grid_square_in_box(n, N, viz = False):
    """
    Create the Laplacian matrix for an n x n grid placed in a larger N x N grid.
    The rest of the grid is set to zero.
    :param n: Grid size (n x n)
    :param N: Larger box grid size (N x N)
    :return: Sparse Laplacian matrix of shape (N^2, N^2)
        """
    # Create an N x N matrix with zeros
    grid_matrix = np.zeros((N, N))

    # Embed the n x n grid (filled with ones) into the center of the N x N matrix
    start_row = (N - n) // 2
    start_col = (N - n) // 2
    grid_matrix[start_row:start_row + n, start_col:start_col + n] = 1

    if viz:
        # Visualize the grid
        plt.figure(figsize=(6, 6))
        plt.imshow(grid_matrix, cmap='Blues', interpolation='none')
        plt.colorbar(label='Matrix Value')
        plt.title(f"{n}x{n} Grid Embedded in {N}x{N} Matrix")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")

        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, N, 1), minor=False)
        ax.set_yticks(np.arange(-0.5, N, 1), minor=False)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        plt.show()

    if n == 1:
        # Special case: 1x1 grid
        L_large = sp.csr_matrix((N * N, N * N))
        center_index = (N // 2) * N + (N // 2)
        L_large[center_index, center_index] = -4
        return L_large

    # Create the Laplacian matrix for the smaller n x n grid
    size = n * n
    diagonals = [-4 * np.ones(size), np.ones(size - 1), np.ones(size - 1), np.ones(size - n), np.ones(size - n)]
    offsets = [0, 1, -1, n, -n]
    L_small = sp.diags(diagonals, offsets, shape=(size, size), format='csr')

    for i in range(1, n):
        L_small[i * n, i * n - 1] = 0
        L_small[i * n - 1, i * n] = 0

    # Embed the smaller Laplacian into a larger N x N grid
    L_large = sp.csr_matrix((N * N, N * N))

    for i in range(n):
        for j in range(n):
            small_idx = i * n + j
            large_idx = (start_row + i) * N + (start_col + j)
            L_large[large_idx, large_idx] = L_small[small_idx, small_idx]
            for k in range(L_small.indptr[small_idx], L_small.indptr[small_idx + 1]):
                neighbor_idx = L_small.indices[k]
                neighbor_large_idx = (start_row + neighbor_idx // n) * N + (start_col + neighbor_idx % n)
                L_large[large_idx, neighbor_large_idx] = L_small.data[k]

    if viz:
        # Visualize the Laplacian matrix
        plt.imshow(L_large.toarray(), cmap='viridis', interpolation='none')
        plt.colorbar(label='Matrix Value')
        plt.title(f"Laplacian Matrix Heatmap for {n}x{n} Grid in {N}x{N} Box")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")

        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, N * N, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N * N, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        plt.show()

    return L_large

def rectangle_laplacian_grid(n, N, viz = False):
    """
    Create the Laplacian matrix for an n x 2n grid using finite difference method.
    :param n: Grid height
    :return: Sparse Laplacian matrix of shape (2n*n, 2n*n)
    """
    m = 2 * n  # Inner grid width
    M = 2 * N  # Container grid width
    size_inner = n * m
    size_container = N * M

    # Create container grid with zeros
    grid_matrix = np.zeros((N, M))
    start_row = (N - n) // 2
    start_col = (M - m) // 2
    grid_matrix[start_row:start_row + n, start_col:start_col + m] = 1

    if viz:
        # Plot the original grid
        plt.figure(figsize=(6, 6))
        plt.imshow(grid_matrix, cmap='Blues', interpolation='none')
        plt.colorbar(label='Matrix Value')
        plt.title(f"{n}x{2*n} Grid Embedded in {N}x{2*N} Container")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        plt.show()

    # Create the Laplacian matrix for the smaller n x 2n grid
    diagonals = [-4 * np.ones(size_inner), np.ones(size_inner - 1), np.ones(size_inner - 1),
                np.ones(size_inner - m), np.ones(size_inner - m)]
    offsets = [0, 1, -1, m, -m]
    L_inner = sp.diags(diagonals, offsets, shape=(size_inner, size_inner), format='csr')

    for i in range(1, n):
        L_inner[i * m, i * m - 1] = 0
        L_inner[i * m - 1, i * m] = 0

    # Embed the smaller Laplacian into the larger N x 2N grid
    L_container = sp.csr_matrix((size_container, size_container))

    for i in range(n):
        for j in range(m):
            small_idx = i * m + j
            large_idx = (start_row + i) * M + (start_col + j)
            L_container[large_idx, large_idx] = L_inner[small_idx, small_idx]
            for k in range(L_inner.indptr[small_idx], L_inner.indptr[small_idx + 1]):
                neighbor_idx = L_inner.indices[k]
                neighbor_large_idx = (start_row + neighbor_idx // m) * M + (start_col + neighbor_idx % m)
                L_container[large_idx, neighbor_large_idx] = L_inner.data[k]

    if viz:
        # Plot the Laplacian matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(L_container.toarray(), cmap='viridis', interpolation='none')
        plt.colorbar(label='Matrix Value')
        plt.title(f"Laplacian Matrix Heatmap for {n}x{2*n} Grid in {N}x{2*N} Container")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, size_container, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size_container, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        plt.show()

    return L_container

def laplacian_circle(n, sparse = True):
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

# eigenfrequencies, eigenmodus = sp.linalg.eigs(laplacian_grid_square(3))
# print(eigenfrequencies)

# Square
Ls = np.array(range(4,50))
results_sq = []
xs_sq = []
for L in Ls:
    eigenfrequencies, eigenmodus = sp.linalg.eigs(laplacian_grid_square_in_box(L,L+1))
    results_sq.append(eigenfrequencies)

for i,res in enumerate(results_sq):
    x = np.full(len(res),Ls[i])
    xs_sq.append(x)
    

plt.scatter(xs_sq,results_sq, s = 10, color = 'green')
plt.title("Square")
plt.show()


# Rectangle
results_rect = []
xs_rect = []
for L in Ls:
    eigenfrequencies, eigenmodus = sp.linalg.eigs(rectangle_laplacian_grid(L, L+1))
    results_rect.append(eigenfrequencies)

for i,res in enumerate(results_rect):
    x = np.full(len(res),Ls[i])
    xs_rect.append(x)
    

plt.scatter(xs_rect,results_rect, s = 10, color = 'blue')
plt.title("Rectangle")
plt.show()

# Circle
results_circ = []
xs_circ = []
for L in Ls:
    laplacian, grid_points = laplacian_circle(L)
    eigenfrequencies, eigenmodus = sp.linalg.eigs(laplacian)
    results_circ.append(eigenfrequencies)

for i,res in enumerate(results_circ):
    x = np.full(len(res),Ls[i])
    xs_circ.append(x)
    

plt.scatter(xs_circ,results_circ, s = 10, color = 'red')
plt.title("Circle")
plt.show()