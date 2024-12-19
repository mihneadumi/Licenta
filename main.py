from performance_profiling.get_system_info import get_cpu_info, get_gpu_info
from performance_profiling.matrix_multiplication.cpu_multi_threaded_prof import run_matrix_mult_cpu_multi_threaded_prof
from performance_profiling.matrix_multiplication.cpu_single_threaded_prof import \
    run_matrix_mult_cpu_single_threaded_prof
from performance_profiling.matrix_multiplication.gpu_acceleration_prof import run_matrix_mult_gpu_acceleration_prof
from performance_profiling.matrix_multiplication.matrix_generation import run_matrix_generation

MATRIX_SIZE = 1000
RUNS = 100

if __name__ == "__main__":
    print(f"[System Info]\nCPU: {get_cpu_info()}\nGPU: {get_gpu_info()}")

    print("Generating matrices...")
    run_matrix_generation(MATRIX_SIZE)
    print("Running Matrix multiplication tests...")
    run_matrix_mult_cpu_single_threaded_prof(RUNS)
    run_matrix_mult_cpu_multi_threaded_prof(RUNS)
    run_matrix_mult_gpu_acceleration_prof(RUNS)

    # same but 10x matrix size
    MATRIX_SIZE = MATRIX_SIZE * 10
    run_matrix_generation(MATRIX_SIZE)
    print("Running Matrix multiplication tests...")
    run_matrix_mult_cpu_single_threaded_prof(RUNS)
    run_matrix_mult_cpu_multi_threaded_prof(RUNS)
    run_matrix_mult_gpu_acceleration_prof(RUNS)


