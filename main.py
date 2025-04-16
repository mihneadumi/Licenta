import time

from constants.params import RUNS, BIG_MATRIX_SIZE, MID_MATRIX_SIZE, SMALL_MATRIX_SIZE, \
    SMALL_ARRAY_LENGTH, MID_ARRAY_LENGTH, BIG_ARRAY_LENGTH
from utils.utils import get_cpu_info, get_gpu_info, get_formatted_elapsed_time, get_ram_info
from performance_profiling.matrix_multiplication.cpu_multi_threaded_prof import run_matrix_mult_cpu_multi_threaded_prof
from performance_profiling.matrix_multiplication.cpu_single_threaded_prof import \
    run_matrix_mult_cpu_single_threaded_prof
from performance_profiling.matrix_multiplication.gpu_acceleration_prof import run_matrix_mult_gpu_acceleration_prof
from performance_profiling.radix_sort.cpu_multi_threaded_prof import run_radix_sort_multi_threaded_prof
from performance_profiling.radix_sort.cpu_single_threaded_prof import run_radix_sort_single_threaded_prof
from performance_profiling.radix_sort.optimised_cpu_single_threaded_prof import \
    run_optimised_radix_sort_single_threaded_prof

def run_matrix_mult_suite(size, runs=RUNS):
    run_matrix_mult_cpu_single_threaded_prof(runs, size)
    run_matrix_mult_cpu_multi_threaded_prof(runs, size)
    run_matrix_mult_gpu_acceleration_prof(runs, size) if hasGPU else print("No CUDA enabled GPU detected, skipping GPU test")

def run_radix_sort_suite(size, runs=RUNS):
    run_radix_sort_single_threaded_prof(runs, size)
    run_radix_sort_multi_threaded_prof(runs, size)
    run_optimised_radix_sort_single_threaded_prof(runs, size)
    # run_radix_sort_gpu_acceleration_prof(runs, size) if hasGPU else print("No CUDA enabled GPU detected, skipping GPU test")

if __name__ == "__main__":
    startTime = time.time()
    print(f"[System Info]\nCPU: {get_cpu_info()}\nGPU: {get_gpu_info()}\nRAM: {get_ram_info()}")
    with open("results/system_info.txt", "w") as f:
        f.write(f"[System Info]\nCPU: {get_cpu_info()}\nGPU: {get_gpu_info()}\nRAM: {get_ram_info()}")
    hasGPU = get_gpu_info() != "No GPU detected"

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running SMALL Matrix multiplication tests...")
    run_matrix_mult_suite(SMALL_MATRIX_SIZE)

    # same but 4x matrix size
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running MID Matrix multiplication tests...")
    run_matrix_mult_suite(MID_MATRIX_SIZE)

    # same but 10x matrix size
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running BIG Matrix multiplication tests...")
    run_matrix_mult_suite(BIG_MATRIX_SIZE, runs=RUNS//2)

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running SMALL Radix Sort tests...")
    run_radix_sort_suite(SMALL_ARRAY_LENGTH)

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running MID Radix Sort tests...")
    run_radix_sort_suite(BIG_ARRAY_LENGTH, runs=RUNS//2)

    print("Total benchmarking time: ", get_formatted_elapsed_time(startTime))
    print("No automatic publishing of results yet, just send me the results folder as zip and I'll handle the ranking ;)")
