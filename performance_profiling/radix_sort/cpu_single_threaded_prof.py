import os
import time
import numpy as np

from algorithms.radix_sort.single_threaded import radix_sort
from utils.utils import write_result_header


def generate_arrays(size, num_arrays):
    """Generates fresh random arrays for each run."""
    return [np.random.randint(0, 10**6, size, dtype=np.int32) for _ in range(num_arrays)]

def profile_radix_sort(arr):
    """Profiles single-threaded Radix Sort."""
    start = time.time()
    radix_sort(arr)
    return time.time() - start


def save_radix_sort_stats(size, runs):
    print("Info: Profiling single-threaded Radix Sort.")

    output_dir = f'results/radix_sort/{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/cpu_single_thread_stats.txt'

    with open(file_path, 'w') as file:
        file.write("Run, Timestamp, Time(s), Data Size, MB/min\n")

        for run_number in range(1, runs + 1):
            arr = np.random.randint(0, 10 ** 6, size, dtype=np.int32)
            sort_time = profile_radix_sort(arr)

            mb_sorted = arr.nbytes / (1024 ** 2)
            mb_per_min = mb_sorted / (sort_time / 60)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            result = f"{run_number}, {timestamp}, {sort_time:.3f}, {mb_per_min:.2f}\n"
            file.write(result)
            print(f"Info: Run {run_number}: {sort_time:.3f} s,{mb_sorted:.2f} MB, {mb_per_min:.2f} MB/min")

def run_radix_sort_single_threaded_prof(runs=10, size=100000):
    save_radix_sort_stats(size, runs)

if __name__ == "__main__":
    run_radix_sort_single_threaded_prof()
