import os
import time
import numpy as np
import cupy as cp

from algorithms.radix_sort.gpu_acceleration import radix_sort_gpu
from constants.string_constants import RESULTS_BASE_PATH, RADIX_SORT_PATH, DATE_FORMAT
from utils.utils import write_result_header

def generate_arrays(size, num_arrays):
    """Generates fresh random arrays for each run."""
    return [np.random.randint(0, 10**6, size, dtype=np.int32) for _ in range(num_arrays)]

def profile_radix_sort_gpu(arr):
    """Profiles GPU-accelerated Radix Sort."""
    arr_gpu = cp.asarray(arr)
    start_gpu = time.time()
    radix_sort_gpu(arr_gpu)
    return time.time() - start_gpu

def save_radix_sort_gpu_stats(size, runs):
    print("Info: Profiling GPU-accelerated Radix Sort.")

    output_dir = f'{RESULTS_BASE_PATH}{RADIX_SORT_PATH}{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/gpu_acceleration_stats.txt'

    with open(file_path, 'w') as file:
        write_result_header(file)

        for run_number in range(1, runs + 1):
            arr = np.random.randint(0, 10**6, size, dtype=np.int32)
            sort_time = profile_radix_sort_gpu(arr)
            timestamp = time.strftime(DATE_FORMAT)
            result = f"{run_number}, {timestamp}, {sort_time:.3f}\n"
            file.write(result)
            print(f"Info: Run {run_number}: {sort_time:.3f} seconds")

def run_radix_sort_gpu_acceleration_prof(runs=10, size=100000):
    save_radix_sort_gpu_stats(size, runs)

if __name__ == "__main__":
    run_radix_sort_gpu_acceleration_prof()
