import os
import time
import numpy as np
from algorithms.matrix_multiplication.gpu_acceleration import gpu_multiply
from constants.string_constants import RESULTS_BASE_PATH, MATRIX_MULTIPLICATION_PATH, DATE_FORMAT
from utils.utils import write_result_header


def generate_matrices(size):
    """Generates fresh matrices for each run."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return A, B

def profile_gpu(A, B):
    """Profiles GPU matrix multiplication."""
    start_gpu = time.time()
    gpu_multiply(A, B)
    return time.time() - start_gpu

def save_gpu_stats(size, runs):
    print("Info: Profiling GPU matrix multiplication.")

    # Define output file path
    output_dir = f'{RESULTS_BASE_PATH}{MATRIX_MULTIPLICATION_PATH}{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/gpu_acceleration_stats.txt'

    # Write header
    with open(file_path, 'w') as file:
        file.write("Run, Timestamp, Time(s), Data Size, MB/min\n")

        # Profile and save results
        for run_number in range(1, runs + 1):
            A, B = generate_matrices(size)  # Generate new matrices for each run
            gpu_multiply_time = profile_gpu(A, B)

            # Calculate data size in MB
            data_size = A.nbytes + B.nbytes  # Total data size (A and B matrices)
            mb_sorted = data_size / (1024 ** 2)  # Convert bytes to MB
            mb_per_min = mb_sorted / (gpu_multiply_time / 60)

            timestamp = time.strftime(DATE_FORMAT)
            result = f"{run_number}, {timestamp}, {gpu_multiply_time:.3f}, {mb_sorted:.2f} MB, {mb_per_min:.2f} MB/min\n"
            file.write(result)
            print(f"Info: Run {run_number}: {gpu_multiply_time:.3f} s, {mb_sorted:.2f} MB, {mb_per_min:.2f} MB/min")

def run_matrix_mult_gpu_acceleration_prof(runs=10, size=1000):
    save_gpu_stats(size, runs)

if __name__ == "__main__":
    run_matrix_mult_gpu_acceleration_prof()
