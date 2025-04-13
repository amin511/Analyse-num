import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method to find the root of a function f.
    
    Parameters:
        f (function): The function for which to find the root.
        df (function): The derivative of the function f.
        x0 (float): Initial guess for the root.
        tol (float): Tolerance for convergence (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).
    
    Returns:
        float: The approximate root.
        list: History of iterations for visualization.
    """
    iterations = []
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            break
        if dfx == 0:
            raise ValueError("Zero derivative. No solution found.")
        x_new = x - fx / dfx
        iterations.append((x, x_new, fx, dfx))
        x = x_new
    return x, iterations

def plot_newton_process(f, iterations):
    """
    Plot the Newton-Raphson iteration process.
    
    Parameters:
        f (function): The function being analyzed.
        iterations (list): History of iterations from newton_raphson.
    """
    # Generate x values for plotting
    x_min = min(it[0] for it in iterations) - 1
    x_max = max(it[0] for it in iterations) + 1
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x) = x² - 4', linewidth=2)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Plot each iteration
    for i, (x, x_new, fx, dfx) in enumerate(iterations):
        # Tangent line: y = dfx*(x - x) + fx
        tangent_line = lambda x_val: dfx * (x_val - x) + fx
        tangent_vals = tangent_line(x_vals)
        
        # Plot tangent line
        plt.plot(x_vals, tangent_vals, '--', alpha=0.7, 
                label=f'Tangente itération {i+1}')
        
        # Plot points and connections
        plt.plot([x, x_new], [fx, 0], 'ro-', markersize=8)
        plt.text(x, fx, f' P{i} (x={x:.2f})', 
                verticalalignment='bottom', horizontalalignment='right')
        plt.text(x_new, 0, f' x{i+1} = {x_new:.4f}', 
                verticalalignment='top' if i%2 else 'bottom')
    
    # Mark the final root
    root = iterations[-1][1]
    plt.plot(root, 0, 'g*', markersize=15, label=f'Racine approchée: {root:.6f}')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Méthode de Newton-Raphson\nRésolution de f(x) = x² - 4 = 0')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define the function and its derivative
    f = lambda x: x**2 - 4  # Function to solve: x² - 4 = 0 (root at x=2)
    df = lambda x: 2*x       # Its derivative: 2x
    
    # Initial guess
    x0 = 3.0
    
    # Apply Newton-Raphson method
    root, iterations = newton_raphson(f, df, x0)
    print(f"Racine approchée: {root:.12f}")
    print(f"Nombre d'itérations: {len(iterations)}")
    
    # Plot the process
    plot_newton_process(f, iterations)
