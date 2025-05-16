import cupy as cp

def gpu_multiply(A, B):
    """Performs GPU-accelerated matrix multiplication."""
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    C_gpu = cp.dot(A_gpu, B_gpu)
    return cp.asnumpy(C_gpu)

# --- Example Usage ---
if __name__ == "__main__":
    A = cp.random.rand(10000, 10000)
    B = cp.random.rand(10000, 10000)
    C = gpu_multiply(A, B)
    print("GPU multiplication complete.")
