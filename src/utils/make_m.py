import numpy as np
import scipy.sparse

def Make_m(L, N):
    # Create grid
    figsize = (8,6)


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
            if (X[i, j]**2 + Y[i, j]**2 > (L/2)**2):
                # Boundary condition for circle
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

    return M