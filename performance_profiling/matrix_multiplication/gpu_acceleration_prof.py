import os
import time
import numpy as np
from algorithms.matrix_multiplication.gpu_acceleration import gpu_multiply
from performance_profiling.get_system_info import get_gpu_info, get_cpu_info

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
    output_dir = f'results/matrix_multiplication/{size}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/gpu_acceleration_stats.txt'

    # Write header
    with open(file_path, 'w') as file:
        file.write(f"# CPU Info: {get_cpu_info()}\n")
        file.write(f"# GPU Info: {get_gpu_info()}\n")
        file.write("# Run Number, Timestamp, Time (s)\n")

        # Profile and save results
        for run_number in range(1, runs + 1):
            A, B = generate_matrices(size)  # Generate new matrices for each run
            gpu_multiply_time = profile_gpu(A, B)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            result = f"{run_number}, {timestamp}, {gpu_multiply_time:.3f}\n"
            file.write(result)
            print(f"Info: Run {run_number}: {gpu_multiply_time:.3f} seconds")

def run_matrix_mult_gpu_acceleration_prof(runs=10, size=1000):
    save_gpu_stats(size, runs)

if __name__ == "__main__":
    run_matrix_mult_gpu_acceleration_prof()
