import time

from constants.params import MATRIX_SIZE, RUNS, ARRAY_LENGTH
from utils.utils import get_cpu_info, get_gpu_info, get_formatted_elapsed_time, get_ram_info
from performance_profiling.matrix_multiplication.cpu_multi_threaded_prof import run_matrix_mult_cpu_multi_threaded_prof
from performance_profiling.matrix_multiplication.cpu_single_threaded_prof import \
    run_matrix_mult_cpu_single_threaded_prof
from performance_profiling.matrix_multiplication.gpu_acceleration_prof import run_matrix_mult_gpu_acceleration_prof
from performance_profiling.radix_sort.cpu_multi_threaded_prof import run_radix_sort_multi_threaded_prof
from performance_profiling.radix_sort.cpu_single_threaded_prof import run_radix_sort_single_threaded_prof
from performance_profiling.radix_sort.optimised_cpu_single_threaded_prof import \
    run_optimised_radix_sort_single_threaded_prof

if __name__ == "__main__":
    startTime = time.time()
    print(f"[System Info]\nCPU: {get_cpu_info()}\nGPU: {get_gpu_info()}\nRAM: {get_ram_info()}")
    with open("results/system_info.txt", "w") as f:
        f.write(f"[System Info]\nCPU: {get_cpu_info()}\nGPU: {get_gpu_info()}\nRAM: {get_ram_info()}")
    hasGPU = get_gpu_info() != "No GPU detected"

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running Matrix multiplication tests...")
    run_matrix_mult_cpu_single_threaded_prof(RUNS, MATRIX_SIZE)
    run_matrix_mult_cpu_multi_threaded_prof(RUNS, MATRIX_SIZE)
    run_matrix_mult_gpu_acceleration_prof(RUNS, MATRIX_SIZE) if hasGPU else print("No CUDA enabled GPU detected, skipping GPU test")

    # same but 10x matrix size
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running BIG Matrix multiplication tests...")
    run_matrix_mult_cpu_single_threaded_prof(RUNS, MATRIX_SIZE*10)
    run_matrix_mult_cpu_multi_threaded_prof(RUNS, MATRIX_SIZE*10)
    run_matrix_mult_gpu_acceleration_prof(RUNS, MATRIX_SIZE*10) if hasGPU else print("No CUDA enabled GPU detected, skipping GPU test")

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running SMALL Radix Sort tests...")
    run_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH//10)
    run_radix_sort_multi_threaded_prof(RUNS, ARRAY_LENGTH//10)
    run_optimised_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH//10)
    # run_radix_sort_gpu_acceleration_prof(RUNS, ARRAY_LENGTH//10) if hasGPU else print("No CUDA enabled GPU detected, skipping GPU test")

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running Radix Sort tests...")
    run_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH)
    run_radix_sort_multi_threaded_prof(RUNS, ARRAY_LENGTH)
    run_optimised_radix_sort_single_threaded_prof(RUNS, ARRAY_LENGTH)
    # run_radix_sort_gpu_acceleration_prof(RUNS, ARRAY_LENGTH) if hasGPU else print("No CUDA enabled GPU detected, skipping GPU test")



    print("No automatic publishing of results yet, just send me the results folder as zip and I'll handle the ranking ;)")
