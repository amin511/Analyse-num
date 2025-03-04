import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def f1(x):
    return x**6 - x - 1

def f2(x):
    return 1 - (1 / 4) * np.cos(x)

def f3(x):
    return np.cos(x) - np.exp(-x)

def f4(x):
    return x**4 - 56.101 * x**3 + 785.6561 * x**2 - 72.7856 * x + 0.078

# Define the roots for each function
roots = {
    "f1(x) = x^6 - x - 1": np.array([-0.778090, 1.134724]),
    "f2(x) = 1 - (1/4) * cos(x)": np.array([]),  # Add roots if known
    "f3(x) = cos(x) - exp(-x)": np.array([0.0, 1.292696]),
    "f4(x) = x^4 - 56.101x^3 + 785.6561x^2 - 72.7856x + 0.078": np.array([0.001084, 0.092172]),
}

# Define the plotting function
def plot_functions():
    x = np.linspace(-2, 2, 1000)
    functions = {
        "f1(x) = x^6 - x - 1": f1,
        "f2(x) = 1 - (1/4) * cos(x)": f2,
        "f3(x) = cos(x) - exp(-x)": f3,
        "f4(x) = x^4 - 56.101x^3 + 785.6561x^2 - 72.7856x + 0.078": f4,
    }
    
    plt.figure(figsize=(12, 10))
    for i, (title, func) in enumerate(functions.items(), 1):
        plt.subplot(2, 2, i)
        plt.plot(x, func(x), label=title)
        plt.axhline(0, color='k', linestyle='--')
        plt.title(title)
        plt.grid(True)
        
        # Plot roots if available
        if roots[title].size > 0:
            plt.plot(roots[title], func(roots[title]), 'ro', label="Roots")
        
        plt.legend()
        print_roots(title, roots[title])
    
    plt.tight_layout()
    plt.show()

# Define the function to print roots
def print_roots(title, roots):
    print(f"\nRoots for {title}:")
    if roots.size > 0:
        for root in roots:
            print(f"Root found at x = {root:.6f}")
    else:
        print("No roots found.")

# Main function to execute the plotting
if __name__ == "__main__":
    plot_functions()
