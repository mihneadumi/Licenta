import numpy as np

def single_threaded_multiply(A, B):
    """Performs single-threaded matrix multiplication."""
    return np.dot(A, B)

# Example usage
if __name__ == "__main__":
    A = np.random.rand(10000, 10000)
    B = np.random.rand(10000, 10000)
    C = single_threaded_multiply(A, B)
    print("Single-threaded multiplication complete.")
