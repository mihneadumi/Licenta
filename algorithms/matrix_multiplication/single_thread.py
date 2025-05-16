import os

import numpy as np

from numba import njit

@njit
def single_threaded_multiply(A, B):
    m, n = A.shape
    nB, p = B.shape

    if n != nB:
        raise ValueError("Number of columns in A must be equal to the number of rows in B")

    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            total = 0.0
            for k in range(n):
                total += A[i, k] * B[k, j]
            C[i, j] = total

    return C

# --- Example Usage ---
if __name__ == "__main__":
    A = np.random.rand(10000, 10000)
    B = np.random.rand(10000, 10000)
    C = single_threaded_multiply(A, B)
    print("Single-threaded multiplication complete.")
