
import scipy.sparse
from scipy.sparse.linalg import eigs, eigsh
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def laplacian(L,N,h): 
    # Create grid
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    # Initialize Laplacian matrix
    size = N * N
    M = scipy.sparse.lil_matrix((size, size), dtype=float)

    # Fill Laplacian matrix
    for i in range(N):
        for j in range(N):
            index = i * N + j  # Convert 2D index to 1D
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                # Boundary condition: v = 0
                M[index, index] = 1
            else:
                # Interior points
                M[index, index] = -4 / h**2
                M[index, index + 1] = 1 / h**2  # Right neighbor
                M[index, index - 1] = 1 / h**2  # Left neighbor
                M[index, index + N] = 1 / h**2  # Top neighbor
                M[index, index - N] = 1 / h**2  # Bottom neighbor

    # Convert to sparse matrix for faster computation
    return M.tocsc()



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

def plot_laplacian(N):
    """
    Plots the Laplacian matrix and the grid structure for an N x N grid.

    Parameters:
        N (int): Number of grid points per dimension.
    """
    # Create grid
    x = np.linspace(0, 4, N)
    y = np.linspace(0, 4, N)
    X, Y = np.meshgrid(x, y)

    # Initialize Laplacian matrix
    size = N * N
    M = scipy.sparse.lil_matrix((size, size), dtype=float)

    # Fill Laplacian matrix
    for i in range(N):
        for j in range(N):
            index = i * N + j  # Convert 2D index to 1D
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                # Boundary condition: v = 0
                M[index, index] = 1
            else:
                # Interior points
                M[index, index] = -4
                M[index, index + 1] = 1  # Right neighbor
                M[index, index - 1] = 1  # Left neighbor
                M[index, index + N] = 1  # Top neighbor
                M[index, index - N] = 1  # Bottom neighbor

    # Convert to dense matrix for visualization
    M_dense = M.toarray()

    # Plot Laplacian matrix
    plt.figure(figsize=(10, 5))

    # Plot 1: Laplacian matrix
    plt.subplot(1, 2, 1)
    plt.imshow(M_dense, cmap='coolwarm', interpolation='none')
    plt.title("Laplacian Matrix")
    plt.colorbar()

    # Plot 2: Grid structure
    plt.subplot(1, 2, 2)
    for i in range(N):
        for j in range(N):
            # Plot grid points
            plt.plot(X[i, j], Y[i, j], 'ko', markersize=10)
            # Plot connections to neighbors
            if i < N - 1:
                plt.plot([X[i, j], X[i + 1, j]], [Y[i, j], Y[i + 1, j]], 'k-')  # Bottom neighbor
            if j < N - 1:
                plt.plot([X[i, j], X[i, j + 1]], [Y[i, j], Y[i, j + 1]], 'k-')  # Right neighbor

    plt.title("Grid Structure")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig("figs/laplacian_and_grid", dpi = 300)
    plt.show()

def plot_original_object(shape, L, N):
    """
    Plots the original object (square, rectangle, or circle) on a grid.

    Parameters:
        shape (str): Shape of the object ("square", "rectangle", or "circle").
        L (float): Side length or diameter of the object.
        N (int): Number of grid points per dimension.
    """
    # Create grid
    if shape == "square":
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
    elif shape == "rectangle":
        x = np.linspace(0, L, N)
        y = np.linspace(0, 2 * L, 2 * N)
    elif shape == "circle":
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # Plot the object
    plt.figure(figsize=(6, 6))
    if shape == "circle":
        # Plot the circle boundary
        circle = plt.Circle((0, 0), L/2, edgecolor='r', facecolor='none', linewidth=2)
        plt.gca().add_patch(circle)
        # Plot grid points inside the circle
        inside_circle = X**2 + Y**2 <= (L/2)**2
        plt.scatter(X[inside_circle], Y[inside_circle], c='b', s=10, label="Grid Points")
    else:
        # Plot grid points for square or rectangle
        plt.scatter(X, Y, c='b', s=10, label="Grid Points")

    plt.title(f"{shape.capitalize()} (L = {L}, N = {N})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_eigenmodes(shape, n, num_modes=6):
    """Solve for the eigenmodes of the specified shape."""
    if shape == 'square':
        L = laplacian_grid_square(n)
        # For visualization
        grid_shape = (n, n)
        reshape_func = lambda v: v.reshape(grid_shape)
    elif shape == 'circle':
        L, points = laplacian_circle(n)
        grid_shape = None
        reshape_func = lambda v: v  # No reshape for circle
    
    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigs(L, k=num_modes, which='SM')
    
    # For wave equation, eigenvalues should be negative
    eigenvalues = -eigenvalues
    
    #extract real part of eigenvalues and eigenvectors
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate frequencies (sqrt of eigenvalues)
    frequencies = np.sqrt(eigenvalues)
    
    return frequencies, eigenvectors, grid_shape, reshape_func, points if shape == 'circle' else None

def plot_eigenmodes(shape, frequencies, eigenvectors, grid_shape, reshape_func, points=None):
    """Plot the first few eigenmodes."""
    num_modes = min(3, len(frequencies))
    
    for i in range(num_modes):
        plt.figure(figsize=(5, 4))
        if shape == 'square':
            mode = reshape_func(eigenvectors[:, i])
            plt.imshow(mode, cmap='RdBu')
            plt.colorbar(label='Amplitude')
        elif shape == 'circle':
            x, y = zip(*points)
            plt.scatter(x, y, c=eigenvectors[:, i], cmap='RdBu', s=30)
            plt.colorbar(label='Amplitude')
            plt.axis('equal')
        
        plt.title(f'Eigenmode {i+1}, λ = {frequencies[i]:.4f}')
        plt.tight_layout()
        plt.savefig(f"figs/eigenmode{i+1}_lambda{frequencies[i]:.4f}_{shape}.png")
        plt.show()

def create_time_animation(shape, frequency, eigenvector, grid_shape, reshape_func, points=None):
    """Create an animation showing time evolution of an eigenmode."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Set up the initial plot
    if shape == 'square':
        mode = reshape_func(eigenvector)
        img = ax.imshow(mode, cmap='RdBu', interpolation='nearest',
                       vmin=-np.abs(eigenvector).max(), vmax=np.abs(eigenvector).max())
        plt.colorbar(img, ax=ax, label='Amplitude')
    elif shape == 'circle':
        x, y = zip(*points)
        scatter = ax.scatter(x, y, c=eigenvector, cmap='RdBu', s=30,
                           vmin=-np.abs(eigenvector).max(), vmax=np.abs(eigenvector).max())
        plt.colorbar(scatter, ax=ax, label='Amplitude')
        ax.set_aspect('equal')
    
    # Define the update function for animation
    def update(frame):
        # Time value between 0 and 2π
        t = 2 * np.pi * frame / 50
        # Apply time dependence: u(x,y,t) = v(x,y) * cos(λt)
        amplitude = np.cos(frequency * t)
        
        if shape == 'square':
            img.set_array(amplitude * reshape_func(eigenvector))
            return [img]
        elif shape == 'circle':
            scatter.set_array(amplitude * eigenvector)
            return [scatter]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=50, interval=100, blit=True)
    
    # Display the title
    ax.set_title(f'Eigenmode Time Evolution, λ = {frequency:.4f}')
    plt.tight_layout()
    
    return anim, fig

if __name__ == "main":
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