import os
import time

from constants.params import RUNS, BIG_MATRIX_SIZE, MID_MATRIX_SIZE, SMALL_MATRIX_SIZE, \
    SMALL_ARRAY_LENGTH, MID_ARRAY_LENGTH, BIG_ARRAY_LENGTH
# Matrix Multiplication Profilers
from performance_profiling.matrix_multiplication.cpu_multi_threaded_prof import run_matrix_mult_cpu_multi_threaded_prof
from performance_profiling.matrix_multiplication.cpu_single_threaded_prof import \
    run_matrix_mult_cpu_single_threaded_prof
from performance_profiling.matrix_multiplication.gpu_acceleration_prof import run_matrix_mult_gpu_acceleration_prof
# Radix Sort Profilers
from performance_profiling.radix_sort.cpu_multi_threaded_prof import run_radix_sort_multi_threaded_prof
from performance_profiling.radix_sort.optimised_cpu_single_threaded_prof import \
    run_optimised_radix_sort_single_threaded_prof
from performance_profiling.radix_sort.gpu_acceleration_prof import run_radix_sort_gpu_acceleration_prof

# K-Means Profiler Imports (Updated)
from performance_profiling.kmeans_clustering.profile_kmeans_all import run_all_kmeans_benchmarks, \
    profile_and_save_stats as profile_kmeans_specific

from utils.utils import get_cpu_info, get_gpu_info, get_formatted_elapsed_time, get_ram_info


def run_matrix_mult_suite(size, runs=RUNS, has_gpu_flag=True):
    print(f"--- Matrix Multiplication Suite: Size {size}, Runs {runs} ---")
    run_matrix_mult_cpu_single_threaded_prof(runs, size)
    run_matrix_mult_cpu_multi_threaded_prof(runs, size)
    if has_gpu_flag:
        run_matrix_mult_gpu_acceleration_prof(runs, size)
    else:
        print(f"Skipping GPU Matrix Multiplication for size {size} (GPU not available/selected).")


def run_radix_sort_suite(size, runs=RUNS, has_gpu_flag=True):
    print(f"--- Radix Sort Suite: Size {size}, Runs {runs} ---")
    run_optimised_radix_sort_single_threaded_prof(runs, size)
    run_radix_sort_multi_threaded_prof(runs, size)
    if has_gpu_flag:
        run_radix_sort_gpu_acceleration_prof(runs, size)
    else:
        print(f"Skipping GPU Radix Sort for size {size} (GPU not available/selected).")


def run_extra_large_tests(runs, has_gpu_flag=True):
    print(f"--- Extra Large Tests Suite: Runs {runs} ---")
    # large_matrix_size = 10000
    # print(f"  Profiling Matrix Multiplication for size {large_matrix_size}...")
    # # For matrix multiplication, single_thread is usually skipped manually for large sizes
    # # by not calling its profiler here for 'large_matrix_size'
    # run_matrix_mult_cpu_multi_threaded_prof(runs, large_matrix_size)
    # if has_gpu_flag:
    #     run_matrix_mult_gpu_acceleration_prof(runs, large_matrix_size)
    # else:
    #     print(f"  Skipping GPU Matrix Multiplication for size {large_matrix_size} (GPU not available/selected).")
    #
    # # Radix Sort - Large
    # extra_large_array_length = BIG_ARRAY_LENGTH * 10  # e.g., 100,000,000 if BIG_ARRAY_LENGTH is 10M
    # print(f"  Profiling Radix Sort for size {extra_large_array_length}...")
    # # Similar to matrix mult, single_thread for radix is usually too slow for this.
    # run_radix_sort_multi_threaded_prof(runs, extra_large_array_length)
    # if has_gpu_flag:
    #     run_radix_sort_gpu_acceleration_prof(runs, extra_large_array_length)
    # else:
    #     print(f"  Skipping GPU Radix Sort for size {extra_large_array_length} (GPU not available/selected).")

    extra_large_kmeans_params = (20 * 1000 * 1000, 32, 10, 50, 1e-4, runs)
    print(f"  Profiling K-Means for {extra_large_kmeans_params[0]} points (skipping single-thread)...")
    profile_kmeans_specific(
        n_points=extra_large_kmeans_params[0],
        n_dims=extra_large_kmeans_params[1],
        n_clusters=extra_large_kmeans_params[2],
        max_iters=extra_large_kmeans_params[3],
        tol=extra_large_kmeans_params[4],
        total_runs=extra_large_kmeans_params[5],
        run_single_thread_impl=False
    )


if __name__ == "__main__":
    startTime = time.time()
    output_dir = "results"
    file_path = os.path.join(output_dir, "system_info.txt")
    os.makedirs(output_dir, exist_ok=True)

    gpu_info_str = "No GPU detected"
    try:
        with open(file_path, "w") as f:
            cpu_info = get_cpu_info()
            gpu_info_str = get_gpu_info()
            ram_info = get_ram_info()
            f.write(f"[System Info]\nCPU: {cpu_info}\nGPU: {gpu_info_str}\nRAM: {ram_info}\n")
        print(f"System info successfully written to {file_path}")
    except IOError as e:
        print(f"Error writing system info to {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred while gathering system info: {e}")

    hasGPU = gpu_info_str != "No GPU detected" and "error" not in gpu_info_str.lower()

    print(f"\nInitial System Check Complete. Elapsed time: {get_formatted_elapsed_time(startTime)}")
    print(f"GPU Available for suites: {hasGPU}")

    # print("\nRunning SMALL Matrix multiplication tests...")
    # run_matrix_mult_suite(SMALL_MATRIX_SIZE, runs=RUNS, has_gpu_flag=hasGPU)
    # print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    # print("Running MID Matrix multiplication tests...")
    # run_matrix_mult_suite(MID_MATRIX_SIZE, runs=RUNS, has_gpu_flag=hasGPU)
    # print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    # print("Running BIG Matrix multiplication tests...")
    # run_matrix_mult_suite(BIG_MATRIX_SIZE, runs=RUNS, has_gpu_flag=hasGPU)
    #
    # print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    # print("Running SMALL Radix Sort tests...")
    # run_radix_sort_suite(SMALL_ARRAY_LENGTH, runs=RUNS, has_gpu_flag=hasGPU)
    # print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    # print("Running MID Radix Sort tests...")
    # run_radix_sort_suite(MID_ARRAY_LENGTH, runs=RUNS, has_gpu_flag=hasGPU)
    # print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    # print("Running BIG Radix Sort tests...")
    # run_radix_sort_suite(BIG_ARRAY_LENGTH, runs=RUNS, has_gpu_flag=hasGPU)
    #
    # # --- K-Means Clustering Standard Tests ---
    # print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    # print("Running K-Means Clustering standard tests...")
    # run_all_kmeans_benchmarks()

    # --- Extra Large Tests (including K-Means without single_thread) ---
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running Extra Large tests...")
    run_extra_large_tests(RUNS, has_gpu_flag=hasGPU)

    print(f"\nTotal benchmarking time: {get_formatted_elapsed_time(startTime)}")
    print(
        "No automatic publishing of results yet, just send me the results folder as zip and I'll handle the ranking ;)")