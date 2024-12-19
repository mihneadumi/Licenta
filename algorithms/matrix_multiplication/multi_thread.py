import numpy as np
from concurrent.futures import ThreadPoolExecutor

def chunk_multiply(start, end, A, B, C):
    """Performs matrix multiplication on a chunk."""
    C[start:end] = np.dot(A[start:end], B)

def multi_threaded_multiply(A, B, num_threads=4):
    """Performs multi-threaded matrix multiplication."""
    C = np.zeros_like(A)
    chunk_size = A.shape[0] // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(chunk_multiply, i, i + chunk_size, A, B, C)
            for i in range(0, A.shape[0], chunk_size)
        ]
        for f in futures:
            f.result()
    return C

# Example usage
if __name__ == "__main__":
    A = np.random.rand(10000, 10000)
    B = np.random.rand(10000, 10000)
    C = multi_threaded_multiply(A, B)
    print("Multi-threaded multiplication complete.")
