import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

def solve_eigenvalue_problem(shape, L, N, plot=True):
    # Create grid
    figsize = (8,6)

    if shape == "square":
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
    elif shape == "rectangle":
        x = np.linspace(0, L, N)
        y = np.linspace(0, 2 * L, 2 * N)
        figsize = (5,6)
    elif shape == "circle":
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
         
    X, Y = np.meshgrid(x, y)

    # Initialize Laplacian matrix
    size = X.size
    M = scipy.sparse.lil_matrix((size, size), dtype=float)

    # Fill Laplacian matrix
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            index = i * X.shape[1] + j  # Convert 2D index to 1D
            if shape == "circle" and (X[i, j]**2 + Y[i, j]**2 > (L/2)**2):
                # Boundary condition for circle
                M[index, index] = 1
            elif i == 0 or i == X.shape[0] - 1 or j == 0 or j == X.shape[1] - 1:
                # Boundary condition for square and rectangle
                M[index, index] = 1
            else:
                # Interior points
                M[index, index] = -4
                M[index, index + 1] = 1  # Right neighbor
                M[index, index - 1] = 1  # Left neighbor
                M[index, index + X.shape[1]] = 1  # Top neighbor
                M[index, index - X.shape[1]] = 1  # Bottom neighbor

    # Convert to sparse matrix for faster computation
    M = M.tocsc()

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(M, k=6, sigma=0)  # Find 6 smallest eigenvalues

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Plot eigenvectors
    for i in range(6):
        v = eigenvectors[:, i].reshape(X.shape)
        plt.figure(figsize=figsize)
        plt.contourf(X, Y, v, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(f"{shape.capitalize()}, Eigenmode {i+1} \n Frequency: {np.sqrt(-eigenvalues[i]):.2f}", fontsize = 28)
        plt.xlabel("x", fontsize = 28)
        plt.ylabel("y", fontsize = 28)
        plt.savefig(f"figs/{str(shape)}_eigenmode{i}", dpi = 300)

        plt.show()
    
    return eigenvalues, eigenvectors
