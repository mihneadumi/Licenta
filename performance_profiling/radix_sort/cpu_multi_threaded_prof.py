import os
import time

import numpy as np

from algorithms.radix_sort.multi_threaded import radix_sort_parallel
from utils.utils import write_result_header


def profile_radix_sort_multi(arr):
    """Profiles multi-threaded Radix Sort."""
    start = time.time()
    radix_sort_parallel(arr)
    return time.time() - start

def save_radix_sort_multi_stats(size, runs):
    print("Info: Profiling multi-threaded Radix Sort.")

    output_dir = f'results/radix_sort/{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/cpu_multi_thread_stats.txt'

    with open(file_path, 'w') as file:
        write_result_header(file)

        for run_number in range(1, runs + 1):
            arr = np.random.randint(0, 10**6, size, dtype=np.int32)  # Generate fresh array each run
            sort_time = profile_radix_sort_multi(arr)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            result = f"{run_number}, {timestamp}, {sort_time:.3f}\n"
            file.write(result)
            print(f"Info: Run {run_number}: {sort_time:.3f} seconds")

def run_radix_sort_multi_threaded_prof(runs=10, size=100000):
    save_radix_sort_multi_stats(size, runs)

if __name__ == "__main__":
    run_radix_sort_multi_threaded_prof()
