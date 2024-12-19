import os
import time
import numpy as np

from algorithms.matrix_multiplication.single_thread import single_threaded_multiply
from performance_profiling.get_system_info import get_cpu_info, get_gpu_info


def profile_cpu_single(A, B):
    """Profiles single-threaded CPU matrix multiplication."""
    start_cpu = time.time()
    single_threaded_multiply(A, B)
    return time.time() - start_cpu

def save_cpu_single_stats(matrix_file, runs):
    print("Info: Profiling single-threaded CPU matrix multiplication.")

    # Load matrices from file
    matrices = np.load(matrix_file)
    A, B = matrices["A"], matrices["B"]
    size = A.shape[0]

    # Define output file path
    output_dir = f'results/matrix_multiplication/{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/cpu_single_thread_stats.txt'

    with open(file_path, 'w') as file:
        file.write(f"# CPU Info: {get_cpu_info()}\n")
        file.write(f"# GPU Info: {get_gpu_info()}\n")
        file.write("# Run Number, Timestamp, Time (s)\n")

        # Profile and save results
        for run_number in range(1, runs + 1):
            cpu_multiply_time = profile_cpu_single(A, B)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            result = f"{run_number}, {timestamp}, {cpu_multiply_time:.3f}\n"
            file.write(result)
            print(f"Info: Run {run_number}: {cpu_multiply_time:.3f} seconds")

def run_matrix_mult_cpu_single_threaded_prof(runs=10):
    matrix_file = "performance_profiling/matrix_multiplication/pre_generated_matrices.npz"
    save_cpu_single_stats(matrix_file, runs)

if __name__ == "__main__":
    run_matrix_mult_cpu_single_threaded_prof()
