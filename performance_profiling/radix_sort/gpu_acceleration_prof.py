import os
import time
import numpy as np
import cupy as cp
import traceback # Import traceback for detailed error printing

# Import the utility function to get GPU info
try:
    from utils.utils import get_gpu_info
except ImportError:
    print("Warning: Could not import 'get_gpu_info' from 'utils.utils'.")
    # Define a fallback function
    def get_gpu_info():
        try:
            return cp.cuda.Device(0).name
        except Exception:
            return "Unknown GPU (Error retrieving name)"

# Attempt to import the GPU radix sort function
try:
    # Ensure this path correctly points to your Python wrapper script
    from algorithms.radix_sort.gpu_acceleration import radix_sort_gpu
except ImportError:
    print("Error: Could not import 'radix_sort_gpu' from 'algorithms.radix_sort.gpu_acceleration'.")
    print("Please ensure the file and function exist and are named correctly.")
    # Define a dummy function if import fails
    def radix_sort_gpu(arr):
        print("Warning: Using dummy radix_sort_gpu function.")
        cp.cuda.Stream.null.synchronize() # Still sync to mimic workload
        return arr # Return the input array

# Define constants for paths and formats
RESULTS_BASE_PATH = 'results/'
RADIX_SORT_PATH = 'radix_sort/' # Specific path for this algorithm
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def profile_radix_sort_gpu(arr_np: np.ndarray):
    """
    Profiles the execution time of the imported GPU radix sort function.

    Args:
        arr_np: NumPy array to be sorted.

    Returns:
        The execution time in seconds, or float('inf') if an error occurs.
    """
    try:
        # Transfer array to GPU
        arr_gpu = cp.asarray(arr_np)
        # Ensure data transfer is complete before starting timer
        cp.cuda.Stream.null.synchronize()

        # Time the GPU sort function call
        start_gpu = time.time()
        # Assuming radix_sort_gpu sorts the array passed to it (in-place or returns sorted)
        sorted_arr_gpu = radix_sort_gpu(arr_gpu)
        # Ensure kernel execution is complete before stopping timer
        cp.cuda.Stream.null.synchronize()
        end_gpu = time.time()

        return end_gpu - start_gpu
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"  CUDA Error during profiling: {e}")
        return float('inf')
    except Exception as e:
        print(f"  Error during GPU profiling: {e}")
        traceback.print_exc() # Print traceback for unexpected errors
        return float('inf')


def save_radix_sort_gpu_stats(size: int, runs: int):
    """
    Runs GPU radix sort profiling for a given size and number of runs,
    saving statistics to a file using MElements/s as the metric.

    Args:
        size: The number of elements in the array to sort.
        runs: The number of times to run the profiling.
    """
    print(f"Info: Profiling GPU-accelerated Radix Sort for size {size:,}. Running {runs} times...")

    # Construct output path using os.path.join
    output_dir = os.path.join(RESULTS_BASE_PATH, RADIX_SORT_PATH, str(size))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'gpu_acceleration_stats.txt')

    try:
        # Check for GPU availability and print name using the utility function
        try:
            gpu_device = cp.cuda.Device(0)
            gpu_device.use()
            # Use the imported get_gpu_info function
            print(f"Info: Using GPU: {get_gpu_info()}")
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error: No CUDA-enabled GPU found or CuPy setup issue: {e}")
            # Write error state to file if GPU check fails
            with open(file_path, 'w') as file:
                # Update header to use MElements/s
                file.write("Run,Timestamp,Time(s),Size,MElements/s\n")
                file.write(f"Error: No CUDA GPU available or CuPy error - {e}\n")
            return # Exit the function if no GPU

        # Open file to write results
        with open(file_path, 'w') as file:
            # Write CSV header - UPDATED METRIC
            file.write("Run,Timestamp,Time(s),Size,MElements/s\n")

            total_time = 0.0
            successful_runs = 0
            for run_number in range(1, runs + 1):
                # Generate fresh random array for each run
                upper_bound = 2**32
                arr_np = np.random.randint(0, upper_bound, size=size, dtype=np.uint32)

                # Profile the GPU sort
                sort_time = profile_radix_sort_gpu(arr_np)

                # Handle potential errors during profiling
                if sort_time == float('inf'):
                    print(f"  Run {run_number}/{runs}: Failed due to error.")
                    timestamp = time.strftime(DATE_FORMAT)
                    # Write error entry to file
                    result = f"{run_number},{timestamp},inf,N/A,N/A\n"
                    file.write(result)
                    continue # Skip to the next run

                # If successful, update statistics
                successful_runs += 1
                total_time += sort_time

                # Calculate performance in Millions of Elements per Second (MElements/s) - UPDATED METRIC
                elements_per_second = size / sort_time if sort_time > 0 else 0
                melements_per_second = elements_per_second / 1e6 # Convert to millions

                # Format results and write to file/console
                timestamp = time.strftime(DATE_FORMAT)
                # Update result line format
                result_line = f"{run_number},{timestamp},{sort_time:.3f},{size},{melements_per_second:.2f}\n"
                file.write(result_line)
                # Update console print - UPDATED METRIC
                print(f"  Run {run_number}/{runs}: {sort_time:.3f} s ({melements_per_second:.2f} MElements/s)")

            # Print average time after all runs
            if successful_runs > 0:
                avg_time = total_time / successful_runs
                print(f"Info: Average sort time over {successful_runs} successful runs: {avg_time:.3f} s")
            else:
                print("Info: No runs completed successfully.")

    except IOError as e:
        print(f"Error writing results to {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during profiling: {e}")
        traceback.print_exc() # Print traceback for unexpected errors


def run_radix_sort_gpu_acceleration_prof(runs: int = 10, size: int = 100000):
    """
    Helper function to start the radix sort profiling.

    Args:
        runs: Number of profiling runs.
        size: Number of elements in the array.
    """
    save_radix_sort_gpu_stats(size, runs)


if __name__ == "__main__":
    # Example usage: Profile sorting 100,000 elements 10 times
    run_radix_sort_gpu_acceleration_prof(runs=10, size=100000)
