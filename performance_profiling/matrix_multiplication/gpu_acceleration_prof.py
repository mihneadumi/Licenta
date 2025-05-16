import os
import time
import numpy as np
import cupy as cp
import traceback

try:
    from utils.utils import get_gpu_info
except ImportError:
    def get_gpu_info():
        try:
            gpu_device = cp.cuda.Device(0)
            name = gpu_device.name
            return name.decode('utf-8') if isinstance(name, bytes) else name
        except Exception:
            return "N/A"

try:
    from algorithms.matrix_multiplication.gpu_acceleration import gpu_multiply
except ImportError:
    print("Error: Could not import 'gpu_multiply' from 'algorithms.matrix_multiplication.gpu_acceleration'.")
    print("Please ensure the file and function exist and are named correctly.")
    def gpu_multiply(A, B):
        print("Warning: Using dummy gpu_multiply function.")
        cp.cuda.Stream.null.synchronize()
        return cp.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)

RESULTS_BASE_PATH = 'results/'
MATRIX_MULTIPLICATION_PATH = 'matrix_multiplication/'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def profile_gpu_multiply(A_np: np.ndarray, B_np: np.ndarray):
    """
    Profiles the execution time of GPU matrix multiplication.

    Args:
        A_np: NumPy array for matrix A.
        B_np: NumPy array for matrix B.

    Returns:
        The execution time in seconds, or float('inf') if an error occurs.
    """
    try:
        A_gpu = cp.asarray(A_np)
        B_gpu = cp.asarray(B_np)
        cp.cuda.Stream.null.synchronize()

        start_gpu = time.time()
        C_gpu = gpu_multiply(A_gpu, B_gpu)
        cp.cuda.Stream.null.synchronize()
        end_gpu = time.time()

        return end_gpu - start_gpu
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"  CUDA Error during profiling: {e}")
        return float('inf')
    except Exception as e:
        print(f"  Error during GPU profiling: {e}")
        traceback.print_exc()
        return float('inf')


def save_matrix_mult_gpu_stats(size: int, runs: int):
    """
    Runs GPU matrix multiplication profiling for a given size and number of runs,
    saving statistics to a file.

    Args:
        size: The dimension of the square matrices (size x size).
        runs: The number of times to run the profiling.
    """
    print(f"Info: Profiling GPU-accelerated Matrix Multiplication for size {size}x{size}. Running {runs} times...")

    output_dir = os.path.join(RESULTS_BASE_PATH, MATRIX_MULTIPLICATION_PATH, str(size))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'gpu_acceleration_stats.txt')

    try:
        try:
            gpu_device = cp.cuda.Device(0)
            gpu_device.use()
            print(f"Info: Using GPU: {get_gpu_info()}")
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error: No CUDA-enabled GPU found or CuPy setup issue: {e}")
            with open(file_path, 'w') as file:
                file.write("Run,Timestamp,Time(s),Data Size (MB),GFLOPS\n")
                file.write(f"Error: No CUDA GPU available or CuPy error - {e}\n")
            return

        with open(file_path, 'w') as file:
            file.write("Run,Timestamp,Time(s),Data Size (MB),GFLOPS\n")

            total_time = 0.0
            successful_runs = 0
            for run_number in range(1, runs + 1):
                A_np = np.random.rand(size, size).astype(np.float64)
                B_np = np.random.rand(size, size).astype(np.float64)
                multiply_time = profile_gpu_multiply(A_np, B_np)

                if multiply_time == float('inf'):
                    print(f"  Run {run_number}/{runs}: Failed due to error.")
                    timestamp = time.strftime(DATE_FORMAT)
                    result = f"{run_number},{timestamp},inf,N/A,N/A\n"
                    file.write(result)
                    continue

                successful_runs += 1
                total_time += multiply_time

                data_size_bytes = A_np.nbytes + B_np.nbytes
                data_size_mb = data_size_bytes / (1024 ** 2)

                gflops = (2.0 * size**3) / (multiply_time * 1e9) if multiply_time > 0 else 0

                timestamp = time.strftime(DATE_FORMAT)
                result_line = f"{run_number},{timestamp},{multiply_time:.3f},{data_size_mb:.2f},{gflops:.2f}\n"
                file.write(result_line)
                print(f"  Run {run_number}/{runs}: {multiply_time:.3f} s ({gflops:.2f} GFLOPS)")

            if successful_runs > 0:
                avg_time = total_time / successful_runs
                print(f"Info: Average multiplication time over {successful_runs} successful runs: {avg_time:.3f} s")
            else:
                print("Info: No runs completed successfully.")

    except IOError as e:
        print(f"Error writing results to {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during profiling: {e}")
        traceback.print_exc()


def run_matrix_mult_gpu_acceleration_prof(runs: int = 10, size: int = 1000):
    """
    Helper function to start the matrix multiplication profiling.

    Args:
        runs: Number of profiling runs.
        size: Dimension of the square matrices.
    """
    save_matrix_mult_gpu_stats(size, runs)


if __name__ == "__main__":
    run_matrix_mult_gpu_acceleration_prof(runs=10, size=1000)