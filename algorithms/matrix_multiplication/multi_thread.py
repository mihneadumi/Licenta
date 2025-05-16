import numpy as np

def multi_threaded_multiply(A, B):
    return np.dot(A, B)

# --- Example Usage ---
if __name__ == "__main__":
    A = np.random.rand(10000, 10000)
    B = np.random.rand(10000, 10000)
    C = multi_threaded_multiply(A, B)
    print("Multi-threaded multiplication complete.")
