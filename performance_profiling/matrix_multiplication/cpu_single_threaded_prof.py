import os
import time
import numpy as np

from algorithms.matrix_multiplication.single_thread import single_threaded_multiply
from constants.string_constants import RESULTS_BASE_PATH, MATRIX_MULTIPLICATION_PATH, DATE_FORMAT
from utils.utils import write_result_header


def generate_matrices(size):
    """Generates fresh matrices for each run."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return A, B

def profile_cpu_single(A, B):
    """Profiles single-threaded CPU matrix multiplication."""
    start_cpu = time.time()
    single_threaded_multiply(A, B)
    return time.time() - start_cpu

def save_cpu_single_stats(size, runs):
    print("Info: Profiling single-threaded CPU matrix multiplication.")

    # Define output file path
    output_dir = f'{RESULTS_BASE_PATH}{MATRIX_MULTIPLICATION_PATH}{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/cpu_single_thread_stats.txt'

    with open(file_path, 'w') as file:
        file.write("Run, Timestamp, Time(s), Data Size, MB/min\n")

        # Profile and save results
        for run_number in range(1, runs + 1):
            A, B = generate_matrices(size)  # Generate new matrices for each run
            cpu_multiply_time = profile_cpu_single(A, B)

            # Calculate data size in MB
            data_size = A.nbytes + B.nbytes  # Total data size (A and B matrices)
            mb_sorted = data_size / (1024 ** 2)  # Convert bytes to MB
            mb_per_min = mb_sorted / (cpu_multiply_time / 60)

            timestamp = time.strftime(DATE_FORMAT)
            result = f"{run_number}, {timestamp}, {cpu_multiply_time:.3f}, {mb_sorted:.2f} MB, {mb_per_min:.2f} MB/min\n"
            file.write(result)
            print(f"Info: Run {run_number}: {cpu_multiply_time:.3f} s, {mb_sorted:.2f} MB, {mb_per_min:.2f} MB/min")

def run_matrix_mult_cpu_single_threaded_prof(runs=10, size=1000):
    save_cpu_single_stats(size, runs)

if __name__ == "__main__":
    run_matrix_mult_cpu_single_threaded_prof()
