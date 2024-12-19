import numpy as np

def save_matrices(filename, size):
    print(f"Info: Generating random matrices of size {size}...")
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    print(f"Info: Saving matrices to {filename}...")
    np.savez(filename, A=A, B=B)
    print("Info: Matrices saved.")

def run_matrix_generation(size=10000):
    save_matrices("performance_profiling/matrix_multiplication/pre_generated_matrices.npz", size)
