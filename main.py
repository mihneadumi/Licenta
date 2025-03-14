from performance_profiling.get_system_info import get_cpu_info, get_gpu_info
from performance_profiling.matrix_multiplication.cpu_multi_threaded_prof import run_matrix_mult_cpu_multi_threaded_prof
from performance_profiling.matrix_multiplication.cpu_single_threaded_prof import \
    run_matrix_mult_cpu_single_threaded_prof
from performance_profiling.matrix_multiplication.gpu_acceleration_prof import run_matrix_mult_gpu_acceleration_prof
from performance_profiling.radix_sort.cpu_multi_threaded_prof import run_radix_sort_multi_threaded_prof
from performance_profiling.radix_sort.cpu_single_threaded_prof import run_radix_sort_single_threaded_prof
from performance_profiling.radix_sort.optimised_cpu_single_threaded_prof import \
    run_optimised_radix_sort_single_threaded_prof

MATRIX_SIZE = 1000
ARRAY_LENGTH = 1000000
RUNS = 2

if __name__ == "__main__":
    print(f"[System Info]\nCPU: {get_cpu_info()}\nGPU: {get_gpu_info()}")

    print("Running Matrix multiplication tests...")
    run_matrix_mult_cpu_single_threaded_prof(RUNS, MATRIX_SIZE)
    run_matrix_mult_cpu_multi_threaded_prof(RUNS, MATRIX_SIZE)
    run_matrix_mult_gpu_acceleration_prof(RUNS, MATRIX_SIZE)

    # same but 10x matrix size
    print("Running BIG Matrix multiplication tests...")
    run_matrix_mult_cpu_single_threaded_prof(RUNS, MATRIX_SIZE*10)
    run_matrix_mult_cpu_multi_threaded_prof(RUNS, MATRIX_SIZE*10)
    run_matrix_mult_gpu_acceleration_prof(RUNS, MATRIX_SIZE*10)

    print("Running Radix Sort tests...")
    run_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH)
    run_radix_sort_multi_threaded_prof(RUNS, ARRAY_LENGTH)
    run_optimised_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH)

    # same but 10x array length
    print("Running BIG Radix Sort tests...")
    run_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH*10)
    run_radix_sort_multi_threaded_prof(RUNS, ARRAY_LENGTH*10)
    run_optimised_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH*10)
