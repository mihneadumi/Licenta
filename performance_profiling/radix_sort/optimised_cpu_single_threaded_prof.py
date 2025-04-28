import os
import time
import numpy as np
import traceback # Import traceback for detailed error printing
import platform # To get CPU info (optional)
import psutil # To get CPU core count (optional, needs installation: pip install psutil)

# Attempt to import the single-threaded CPU radix sort function
try:
    # Using the function name from the provided script
    from algorithms.radix_sort.optimised_radix_sort import radix_sort
except ImportError:
    print("Error: Could not import 'radix_sort' from 'algorithms.radix_sort.optimised_radix_sort'.")
    print("Please ensure the file and function exist and are named correctly.")
    # Define a dummy function to avoid crashing
    def radix_sort(arr):
        print("Warning: Using dummy radix_sort function.")
        # Mimic in-place sort if dummy is used
        arr.sort()
        return arr # Or None if original returns None

# Define constants for paths and formats
RESULTS_BASE_PATH = 'results/'
RADIX_SORT_PATH = 'radix_sort/' # Specific path for this algorithm
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def profile_radix_sort(arr_np: np.ndarray):
    """
    Profiles the execution time of the imported single-threaded radix sort function.

    Args:
        arr_np: NumPy array to be sorted. The sort might happen in-place.

    Returns:
        The execution time in seconds, or float('inf') if an error occurs.
    """
    try:
        # Time the CPU single-threaded sort function call
        start_cpu = time.time()
        # Assuming radix_sort sorts the array passed to it (likely in-place)
        radix_sort(arr_np)
        end_cpu = time.time()
        # No explicit sync needed for CPU timing

        return end_cpu - start_cpu
    except Exception as e:
        print(f"  Error during CPU single-threaded profiling: {e}")
        traceback.print_exc() # Print traceback for unexpected errors
        return float('inf')


def save_radix_sort_stats(size: int, runs: int):
    """
    Runs single-threaded CPU radix sort profiling for a given size and number of runs,
    saving statistics to a file using MElements/s as the metric.

    Args:
        size: The number of elements in the array to sort.
        runs: The number of times to run the profiling.
    """
    print(f"Info: Profiling Single-threaded CPU Radix Sort for size {size:,}. Running {runs} times...") # Matched style

    # Construct output path using os.path.join
    output_dir = os.path.join(RESULTS_BASE_PATH, RADIX_SORT_PATH, str(size))
    os.makedirs(output_dir, exist_ok=True)
    # Use the specific file name from the user's script context
    file_path = os.path.join(output_dir, 'optimised_cpu_single_thread_stats.txt')

    try:
        # Print CPU information (optional but helpful)
        try:
            print(f"Info: Using CPU: {platform.processor()}")
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            print(f"Info: CPU Cores: {physical_cores} physical, {logical_cores} logical (Note: single-threaded uses only one core)")
        except Exception as e:
            print(f"Info: Could not retrieve detailed CPU info ({e}).")

        # Open file to write results
        with open(file_path, 'w') as file:
            # Write CSV header - UPDATED METRIC
            file.write("Run,Timestamp,Time(s),Size,MElements/s\n")

            total_time = 0.0
            successful_runs = 0
            for run_number in range(1, runs + 1):
                # Generate fresh random uint32 array for each run
                upper_bound = 2**32
                arr_np = np.random.randint(0, upper_bound, size=size, dtype=np.uint32)
                # Create a copy if the sort function modifies in-place and you need original later
                arr_to_sort = arr_np.copy()

                # Profile the CPU single-threaded sort
                sort_time = profile_radix_sort(arr_to_sort)

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


def run_optimised_radix_sort_single_threaded_prof(runs: int = 10, size: int = 100000):
    """
    Helper function to start the single-threaded CPU radix sort profiling.

    Args:
        runs: Number of profiling runs.
        size: Number of elements in the array.
    """
    save_radix_sort_stats(size, runs)


if __name__ == "__main__":
    # Example usage: Profile sorting 100,000 elements 10 times
    run_optimised_radix_sort_single_threaded_prof(runs=10, size=100000)
