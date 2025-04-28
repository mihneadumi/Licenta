import os
import time
import numpy as np
import traceback # Import traceback for detailed error printing
import platform # To get CPU info (optional)
import psutil # To get CPU core count (optional, needs installation: pip install psutil)

# Attempt to import the single-threaded CPU matrix multiplication function
try:
    # Adjust path if necessary
    from algorithms.matrix_multiplication.single_thread import single_threaded_multiply
except ImportError:
    print("Error: Could not import 'single_threaded_multiply' from 'algorithms.matrix_multiplication.single_thread'.")
    print("Please ensure the file and function exist and are named correctly.")
    # Define a dummy function if import fails
    def single_threaded_multiply(A, B):
        print("Warning: Using dummy single_threaded_multiply function.")
        # Mimic workload roughly - use numpy's matmul or return zeros.
        # Returning zeros is safer for profiling comparison if the dummy is used.
        return np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)

# Define constants for paths and formats
RESULTS_BASE_PATH = 'results/'
MATRIX_MULTIPLICATION_PATH = 'matrix_multiplication/' # Specific path for this algorithm
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def generate_matrices(size):
    """Generates fresh matrices for each run."""
    # Using float64 for consistency with other profilers
    A = np.random.rand(size, size).astype(np.float64)
    B = np.random.rand(size, size).astype(np.float64)
    return A, B

def profile_cpu_single(A_np: np.ndarray, B_np: np.ndarray):
    """
    Profiles the execution time of single-threaded CPU matrix multiplication.

    Args:
        A_np: NumPy array for matrix A.
        B_np: NumPy array for matrix B.

    Returns:
        The execution time in seconds, or float('inf') if an error occurs.
    """
    try:
        # Time the CPU single-threaded multiplication
        start_cpu = time.time()
        C_np = single_threaded_multiply(A_np, B_np)
        end_cpu = time.time()
        # No explicit sync needed for CPU timing

        # Optional: Verify result correctness here if needed, outside timing

        return end_cpu - start_cpu
    except Exception as e:
        print(f"  Error during CPU single-threaded profiling: {e}")
        traceback.print_exc() # Print traceback for unexpected errors
        return float('inf')

def save_cpu_single_stats(size: int, runs: int):
    """
    Runs single-threaded CPU matrix multiplication profiling for a given size
    and number of runs, saving statistics to a file using GFLOPS as the metric.

    Args:
        size: The dimension of the square matrices (size x size).
        runs: The number of times to run the profiling.
    """
    print(f"Info: Profiling Single-threaded CPU Matrix Multiplication for size {size}x{size}. Running {runs} times...")

    # Construct output path using os.path.join
    output_dir = os.path.join(RESULTS_BASE_PATH, MATRIX_MULTIPLICATION_PATH, str(size))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'cpu_single_thread_stats.txt') # Specific filename

    try:
        # Print CPU information (optional but helpful)
        try:
            print(f"Info: Using CPU: {platform.processor()}")
            # Requires psutil: pip install psutil
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            print(f"Info: CPU Cores: {physical_cores} physical, {logical_cores} logical (Note: single-threaded uses only one core)")
        except Exception as e:
            print(f"Info: Could not retrieve detailed CPU info ({e}).")

        # Open file to write results
        with open(file_path, 'w') as file:
            # Write CSV header - Use GFLOPS
            file.write("Run,Timestamp,Time(s),Data Size (MB),GFLOPS\n")

            total_time = 0.0
            successful_runs = 0
            for run_number in range(1, runs + 1):
                # Generate fresh random matrices for each run
                A_np, B_np = generate_matrices(size)

                # Profile the CPU single-threaded multiplication
                multiply_time = profile_cpu_single(A_np, B_np)

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
                data_size_bytes = A_np.nbytes + B_np.nbytes
                data_size_mb = data_size_bytes / (1024 ** 2)

                # Calculate performance in GFLOPS
                gflops = (2.0 * size**3) / (multiply_time * 1e9) if multiply_time > 0 else 0

                # Format results and write to file/console
                timestamp = time.strftime(DATE_FORMAT)
                result_line = f"{run_number},{timestamp},{multiply_time:.3f},{data_size_mb:.2f},{gflops:.2f}\n"
                file.write(result_line)
                # Print run result to console, matching other profilers
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


def run_matrix_mult_cpu_single_threaded_prof(runs: int = 10, size: int = 1000):
    """
    Helper function to start the single-threaded CPU matrix multiplication profiling.

    Args:
        runs: Number of profiling runs.
        size: Dimension of the square matrices.
    """
    save_cpu_single_stats(size, runs)


if __name__ == "__main__":
    # Example usage: Profile 1000x1000 matrix multiplication 10 times
    run_matrix_mult_cpu_single_threaded_prof(runs=10, size=1000)
