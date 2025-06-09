import os
import time
import argparse

from constants.params import RUNS, BIG_MATRIX_SIZE, MID_MATRIX_SIZE, SMALL_MATRIX_SIZE, \
    SMALL_ARRAY_LENGTH, MID_ARRAY_LENGTH, BIG_ARRAY_LENGTH
from performance_profiling.kmeans_clustering.profile_kmeans_all import run_all_kmeans_benchmarks
from performance_profiling.matrix_multiplication.profile_matrix_mult_all import run_matrix_multiplication_benchmark
from performance_profiling.radix_sort.profile_radix_sort_all import run_radix_sort_benchmark
from utils.utils import get_cpu_info, get_gpu_info, get_formatted_elapsed_time, get_ram_info


def run_matrix_mult_suite(size, runs=RUNS, has_gpu_flag=True):
    print(f"--- Matrix Multiplication Suite: Size {size}, Runs {runs} ---")
    run_matrix_multiplication_benchmark(
        size=size,
        runs=runs,
        run_single_thread=True,
        run_gpu=has_gpu_flag
    )


def run_radix_sort_suite(size, runs=RUNS, has_gpu_flag=True):
    print(f"--- Radix Sort Suite: Size {size}, Runs {runs} ---")
    run_radix_sort_benchmark(
        size=size,
        runs=runs,
        run_single_thread=True,
        run_gpu=has_gpu_flag
    )


def run_extra_large_tests(runs, has_gpu_flag=True):
    print(f"--- Extra Large Tests Suite: Runs {runs} ---")

    large_matrix_size = 10000
    print(f"  Profiling Matrix Multiplication for size {large_matrix_size} (single-thread will be skipped)...")
    run_matrix_multiplication_benchmark(
        size=large_matrix_size,
        runs=runs,
        run_single_thread=True,
        run_gpu=has_gpu_flag
    )

    extra_large_array_length = BIG_ARRAY_LENGTH * 10
    print(f"  Profiling Radix Sort for size {extra_large_array_length} (single-thread will be skipped)...")
    run_radix_sort_benchmark(
        size=extra_large_array_length,
        runs=runs,
        run_single_thread=True,
        run_gpu=has_gpu_flag
    )


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(description="Main benchmark runner script.")
    main_parser.add_argument(
        "--skip_gpu_all",
        action="store_true",
        help="If set, skips all GPU tests throughout the script."
    )
    main_args = main_parser.parse_args()

    startTime = time.time()
    output_dir = "results"
    file_path = os.path.join(output_dir, "system_info.txt")
    os.makedirs(output_dir, exist_ok=True)

    gpu_info_str = "No GPU detected"
    try:
        with open(file_path, "w") as f:
            cpu_info = get_cpu_info()
            if not main_args.skip_gpu_all:
                gpu_info_str = get_gpu_info()
            else:
                gpu_info_str = "GPU tests globally skipped by user."
            ram_info = get_ram_info()
            f.write(f"[System Info]\nCPU: {cpu_info}\nGPU: {gpu_info_str}\nRAM: {ram_info}\n")
        print(f"System info successfully written to {file_path}")
    except IOError as e:
        print(f"Error writing system info to {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred while gathering system info: {e}")

    hasGPU_system_detected = gpu_info_str != "No GPU detected" and \
                             "error" not in gpu_info_str.lower() and \
                             "skipped" not in gpu_info_str.lower()
    run_gpu_tests_globally = hasGPU_system_detected and not main_args.skip_gpu_all

    print(f"\nInitial System Check Complete. Elapsed time: {get_formatted_elapsed_time(startTime)}")
    print(f"GPU Available for suites (based on detection and flags): {run_gpu_tests_globally}")

    print("\nRunning SMALL Matrix multiplication tests...")
    run_matrix_mult_suite(SMALL_MATRIX_SIZE, runs=RUNS, has_gpu_flag=run_gpu_tests_globally)
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running MID Matrix multiplication tests...")
    run_matrix_mult_suite(MID_MATRIX_SIZE, runs=RUNS, has_gpu_flag=run_gpu_tests_globally)
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running BIG Matrix multiplication tests...")
    run_matrix_mult_suite(BIG_MATRIX_SIZE, runs=RUNS, has_gpu_flag=run_gpu_tests_globally)

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running SMALL Radix Sort tests...")
    run_radix_sort_suite(SMALL_ARRAY_LENGTH, runs=RUNS, has_gpu_flag=run_gpu_tests_globally)
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running MID Radix Sort tests...")
    run_radix_sort_suite(MID_ARRAY_LENGTH, runs=RUNS, has_gpu_flag=run_gpu_tests_globally)
    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running BIG Radix Sort tests...")
    run_radix_sort_suite(BIG_ARRAY_LENGTH, runs=RUNS, has_gpu_flag=run_gpu_tests_globally)

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running K-Means Clustering standard tests...")
    run_all_kmeans_benchmarks(include_single_thread_for_standard_tests=True)

    print(f"\nElapsed time: {get_formatted_elapsed_time(startTime)}")
    print("Running Extra Large tests...")
    run_extra_large_tests(6, has_gpu_flag=run_gpu_tests_globally)

    print(f"\nTotal benchmarking time: {get_formatted_elapsed_time(startTime)}")
    print(
        "No automatic publishing of results yet, just send me the results folder as zip and I'll handle the ranking ;)")