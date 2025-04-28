import os
import time
import numpy as np
import cupy as cp
import traceback # Import traceback for detailed error printing

from utils.utils import get_gpu_info

# Attempt to import the GPU matrix multiplication function
try:
    # Make sure the path is correct relative to where this script is run
    # Or ensure the 'algorithms' directory is in the Python path
    from algorithms.matrix_multiplication.gpu_acceleration import gpu_multiply
except ImportError:
    print("Error: Could not import 'gpu_multiply' from 'algorithms.matrix_multiplication.gpu_acceleration'.")
    print("Please ensure the file and function exist and are named correctly.")
    # Define a dummy function if import fails, allowing the script to run partially
    def gpu_multiply(A, B):
        print("Warning: Using dummy gpu_multiply function.")
        cp.cuda.Stream.null.synchronize() # Still sync to mimic workload
        # Return a dummy result of the expected type (CuPy array) and shape
        return cp.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)

# Define constants for paths and formats
RESULTS_BASE_PATH = 'results/'
MATRIX_MULTIPLICATION_PATH = 'matrix_multiplication/' # Specific path for this algorithm
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
        # Transfer matrices to GPU
        A_gpu = cp.asarray(A_np)
        B_gpu = cp.asarray(B_np)
        # Ensure data transfers are complete before starting timer
        cp.cuda.Stream.null.synchronize()

        # Time the GPU multiplication
        start_gpu = time.time()
        # Assuming gpu_multiply performs C = A @ B and returns C,
        # or modifies a pre-allocated C. The timing includes kernel execution.
        C_gpu = gpu_multiply(A_gpu, B_gpu)
        # Ensure kernel execution is complete before stopping timer
        cp.cuda.Stream.null.synchronize()
        end_gpu = time.time()

        # Optional: Verify result correctness here if needed, outside timing

        return end_gpu - start_gpu
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"  CUDA Error during profiling: {e}")
        return float('inf')
    except Exception as e:
        print(f"  Error during GPU profiling: {e}")
        traceback.print_exc() # Print traceback for unexpected errors
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

    # Construct output path using os.path.join for cross-platform compatibility
    output_dir = os.path.join(RESULTS_BASE_PATH, MATRIX_MULTIPLICATION_PATH, str(size))
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    file_path = os.path.join(output_dir, 'gpu_acceleration_stats.txt')

    try:
        # Check for GPU availability and print name
        try:
            gpu_device = cp.cuda.Device(0)
            gpu_device.use()
            print(f"Info: Using GPU: {get_gpu_info()}") # Use attribute for name
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error: No CUDA-enabled GPU found or CuPy setup issue: {e}")
            # Write error state to file if GPU check fails
            with open(file_path, 'w') as file:
                # Match header, use GFLOPS for matmul
                file.write("Run,Timestamp,Time(s),Data Size (MB),GFLOPS\n")
                file.write(f"Error: No CUDA GPU available or CuPy error - {e}\n")
            return # Exit the function if no GPU

        # Open file to write results
        with open(file_path, 'w') as file:
            # Write CSV header - Use GFLOPS for matrix multiplication
            file.write("Run,Timestamp,Time(s),Data Size (MB),GFLOPS\n")

            total_time = 0.0
            successful_runs = 0
            for run_number in range(1, runs + 1):
                # Generate fresh random matrices for each run
                # Using float64 for potentially higher precision needs in matmul, adjust if needed
                A_np = np.random.rand(size, size).astype(np.float64)
                B_np = np.random.rand(size, size).astype(np.float64)

                # Profile the GPU multiplication
                multiply_time = profile_gpu_multiply(A_np, B_np)

                # Handle potential errors during profiling
                if multiply_time == float('inf'):
                    print(f"  Run {run_number}/{runs}: Failed due to error.")
                    timestamp = time.strftime(DATE_FORMAT)
                    # Write error entry to file
                    result = f"{run_number},{timestamp},inf,N/A,N/A\n"
                    file.write(result)
                    continue # Skip to the next run

                # If successful, update statistics
                successful_runs += 1
                total_time += multiply_time

                # Calculate input data size in MB
                # (Consider if output matrix C should be included depending on analysis goal)
                data_size_bytes = A_np.nbytes + B_np.nbytes
                data_size_mb = data_size_bytes / (1024 ** 2)

                # Calculate performance in GFLOPS (Giga Floating Point Operations Per Second)
                # For standard matrix multiplication C=A*B (N*N matrices), operations = 2*N^3 - N^2 ≈ 2*N^3
                gflops = (2.0 * size**3) / (multiply_time * 1e9) if multiply_time > 0 else 0

                # Format results and write to file/console
                timestamp = time.strftime(DATE_FORMAT)
                result_line = f"{run_number},{timestamp},{multiply_time:.3f},{data_size_mb:.2f},{gflops:.2f}\n"
                file.write(result_line)
                # Print run result to console, matching radix sort style
                print(f"  Run {run_number}/{runs}: {multiply_time:.3f} s ({gflops:.2f} GFLOPS)")

            # Print average time after all runs
            if successful_runs > 0:
                avg_time = total_time / successful_runs
                print(f"Info: Average multiplication time over {successful_runs} successful runs: {avg_time:.3f} s")
            else:
                print("Info: No runs completed successfully.")

    except IOError as e:
        print(f"Error writing results to {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during profiling: {e}")
        traceback.print_exc() # Print traceback for unexpected errors


def run_matrix_mult_gpu_acceleration_prof(runs: int = 10, size: int = 1000):
    """
    Helper function to start the matrix multiplication profiling.

    Args:
        runs: Number of profiling runs.
        size: Dimension of the square matrices.
    """
    save_matrix_mult_gpu_stats(size, runs)


if __name__ == "__main__":
    # Example usage: Profile 1000x1000 matrix multiplication 10 times
    run_matrix_mult_gpu_acceleration_prof(runs=10, size=1000)
