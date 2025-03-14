import os
import time

from algorithms.matrix_multiplication.multi_thread import multi_threaded_multiply
from constants.string_constants import RESULTS_BASE_PATH, MATRIX_MULTIPLICATION_PATH, DATE_FORMAT
from performance_profiling.utils import get_cpu_info, get_gpu_info, generate_matrices, write_result_header


def profile_cpu_multi(A, B):
    """Profiles multi-threaded CPU matrix multiplication."""
    start_cpu = time.time()
    multi_threaded_multiply(A, B)
    return time.time() - start_cpu

def save_cpu_multi_stats(size, runs):
    print("Info: Profiling multi-threaded CPU matrix multiplication.")

    # Define output file path
    output_dir = f'{RESULTS_BASE_PATH}{MATRIX_MULTIPLICATION_PATH}{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/cpu_multi_thread_stats.txt'

    with open(file_path, 'w') as file:
        write_result_header(file)

        # Profile and save results
        for run_number in range(1, runs + 1):
            A, B = generate_matrices(size)  # Generate new matrices for each run
            cpu_multiply_time = profile_cpu_multi(A, B)
            timestamp = time.strftime(DATE_FORMAT)
            result = f"{run_number}, {timestamp}, {cpu_multiply_time:.3f}\n"
            file.write(result)
            print(f"Info: Run {run_number}: {cpu_multiply_time:.3f} seconds")

def run_matrix_mult_cpu_multi_threaded_prof(runs=10, size=1000):
    save_cpu_multi_stats(size, runs)

if __name__ == "__main__":
    run_matrix_mult_cpu_multi_threaded_prof()
