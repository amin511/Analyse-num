
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def gauss_elimination(A, b, visualize=True):
    """
    Solve Ax = b with Gaussian elimination + visualization
    Args:
        A: n x n coefficient matrix
        b: n x 1 right-hand side vector
        visualize: whether to show matrix transformations
    Returns:
        x: solution vector
        steps: list of intermediate matrices
    """
    n = len(b)
    Ab = np.hstack([A.astype(float), b.astype(float).reshape(-1, 1)])
    steps = [Ab.copy()]
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#66b3ff', '#0000ff'], N=256)
    
    if visualize:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(Ab[:, :-1], cmap=cmap, vmin=-10, vmax=10)
        plt.colorbar()
        plt.title("Initial Matrix A")
        
        plt.subplot(1, 2, 2)
        plt.imshow(Ab[:, -1].reshape(-1, 1), cmap='Reds')
        plt.colorbar()
        plt.title("Vector b")
        plt.suptitle("Initial System")
        plt.show()
    
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        steps.append(Ab.copy())
        
        # Elimination
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
            steps.append(Ab.copy())
        
        if visualize and i < n-1:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(Ab[:, :-1], cmap=cmap, vmin=-10, vmax=10)
            plt.colorbar()
            plt.title(f"Matrix after step {i+1}")
            
            plt.subplot(1, 2, 2)
            plt.imshow(Ab[:, -1].reshape(-1, 1), cmap='Reds')
            plt.colorbar()
            plt.title("Updated vector b")
            plt.suptitle(f"Elimination step {i+1}")
            plt.show()
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    if visualize:
        # Plot final triangular system
        plt.figure(figsize=(8, 6))
        plt.imshow(Ab[:, :-1], cmap=cmap, vmin=-10, vmax=10)
        plt.colorbar()
        plt.title("Final Upper Triangular Matrix")
        plt.show()
        
        # Plot solution
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, n+1), x, color='green', alpha=0.6)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.xlabel('Variable index')
        plt.ylabel('Solution value')
        plt.title('Solution Vector x')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    return x, steps

# Numerical application from Exercise 1
A = np.array([
    [2, -1, 4, 0],
    [4, -1, 5, 1],
    [-2, 2, -2, 3],
    [0, 3, -9, 4]
])

b = np.array([1, 0, 0, 0])

# Solve with visualization
solution, _ = gauss_elimination(A, b)

print("\nFinal solution:")
for i, val in enumerate(solution):
    print(f"x_{i+1} = {val:.6f}")
