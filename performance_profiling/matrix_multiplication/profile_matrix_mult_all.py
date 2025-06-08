import os
import time
import numpy as np
import cupy as cp
import traceback
import platform
import psutil
import argparse

from algorithms.matrix_multiplication.single_thread import single_threaded_multiply
from algorithms.matrix_multiplication.multi_thread import multi_threaded_multiply
from algorithms.matrix_multiplication.gpu_acceleration import gpu_multiply
from utils.utils import get_gpu_info

RESULTS_BASE_PATH = 'results/'
MATRIX_MULTIPLICATION_PATH = 'matrix_multiplication/'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RANDOM_SEED = 42
DATA_TYPE = np.float64


def generate_matrices(matrix_dim, random_state_seed):
    rng = np.random.RandomState(random_state_seed)
    A_np = rng.rand(matrix_dim, matrix_dim).astype(DATA_TYPE)
    B_np = rng.rand(matrix_dim, matrix_dim).astype(DATA_TYPE)
    return A_np, B_np


def profile_and_save_stats(
        matrix_dim: int,
        total_runs: int,
        run_single_thread_impl: bool = True,
        run_gpu_impl: bool = True
):
    size_str = f"N{matrix_dim}"
    print(f"\nInfo: Profiling Matrix Multiplication for configuration: {size_str} ({matrix_dim}x{matrix_dim})")
    print(f"Parameters: Runs={total_runs}, Data Type={DATA_TYPE.__name__}")
    if not run_single_thread_impl:
        print("  NOTE: Single-threaded CPU implementation will be SKIPPED for this configuration.")
    if not run_gpu_impl:
        print("  NOTE: GPU implementation will be SKIPPED for this configuration.")

    output_dir = os.path.join(RESULTS_BASE_PATH, MATRIX_MULTIPLICATION_PATH, str(matrix_dim))
    os.makedirs(output_dir, exist_ok=True)

    impl_config = {
        "single_thread": {
            "file_suffix": 'cpu_single_thread_stats.txt',
            "func": single_threaded_multiply,
            "run_this_time": run_single_thread_impl,
            "is_gpu": False,
            "name_print": "CPU Single-Thread MM"
        },
        "multi_thread": {
            "file_suffix": 'cpu_multi_thread_stats.txt',
            "func": multi_threaded_multiply,
            "run_this_time": True,
            "is_gpu": False,
            "name_print": "CPU Multi-Thread MM"
        },
        "gpu_custom": {
            "file_suffix": 'gpu_acceleration_stats.txt',
            "func": gpu_multiply,
            "run_this_time": run_gpu_impl,
            "is_gpu": True,
            "name_print": "GPU Custom MM (CuPy/cuBLAS)"
        }
    }

    file_handles = {}
    active_implementations_for_run = {}

    try:
        for key, config_item in impl_config.items():
            if config_item["run_this_time"]:
                path = os.path.join(output_dir, config_item["file_suffix"])
                file_handles[key] = open(path, 'w')
                file_handles[key].write("Run,Timestamp,Time(s),Data Size (MB),GFLOPS\n")
                active_implementations_for_run[key] = config_item

        if not active_implementations_for_run:
            print(f"    No Matrix Multiplication implementations selected to run for {size_str}. Skipping.")
            return

        for run_number in range(1, total_runs + 1):
            print(f"  Starting Run {run_number}/{total_runs} for {size_str}...")
            current_run_seed = RANDOM_SEED + run_number

            A_np, B_np = generate_matrices(matrix_dim, current_run_seed)

            data_size_bytes_per_matrix = A_np.nbytes
            data_size_mb_total = (data_size_bytes_per_matrix * 2) / (1024 ** 2)

            for impl_key, config_item in active_implementations_for_run.items():
                func_to_profile = config_item["func"]
                impl_name_print = config_item["name_print"]
                is_gpu = config_item["is_gpu"]

                A_input = cp.asarray(A_np) if is_gpu else A_np.copy()
                B_input = cp.asarray(B_np) if is_gpu else B_np.copy()

                print(f"    Profiling {impl_name_print}...")
                exec_time = float('inf')
                gflops = 0.0

                try:
                    if is_gpu:
                        cp.cuda.Stream.null.synchronize()

                    start_time = time.perf_counter()
                    _ = func_to_profile(A_input, B_input)

                    if is_gpu:
                        cp.cuda.Stream.null.synchronize()
                    end_time = time.perf_counter()
                    exec_time = end_time - start_time

                    if exec_time > 0:
                        gflops = (2.0 * matrix_dim ** 3) / (exec_time * 1e9)

                    timestamp = time.strftime(DATE_FORMAT)
                    result_line = (f"{run_number},{timestamp},{exec_time:.6f},"
                                   f"{data_size_mb_total:.2f},{gflops:.2f}\n")
                    file_handles[impl_key].write(result_line)
                    print(
                        f"      {impl_name_print} Run {run_number}: {exec_time:.6f}s, "
                        f"GFLOPS: {gflops:.2f}")
                except Exception as e:
                    print(f"      Error during {impl_name_print} profiling for run {run_number}: {e}")
                    traceback.print_exc()
                    timestamp = time.strftime(DATE_FORMAT)
                    result_line = (f"{run_number},{timestamp},inf,N/A,N/A\n")
                    if impl_key in file_handles:
                        file_handles[impl_key].write(result_line)
        print(f"  Finished all runs for {size_str}.")
    except IOError as e_io:
        print(f"Error writing results for {size_str}: {e_io}")
    except Exception as e_outer:
        print(f"An unexpected error occurred during profiling for {size_str}: {e_outer}")
        traceback.print_exc()
    finally:
        for fh_name, fh in file_handles.items():
            if fh and not fh.closed:
                print(f"    Closing file for {fh_name}.")
                fh.close()


def run_matrix_multiplication_benchmark(size: int, runs: int, run_single_thread: bool = True, run_gpu: bool = True):
    try:
        print(f"CPU Info: {platform.processor()}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    except Exception as e_cpu_info:
        print(f"Could not get CPU info: {e_cpu_info}")

    if run_gpu:
        try:
            cp.cuda.Device(0).use()
            print(f"GPU Info: {get_gpu_info()}")
        except Exception as e_gpu_info_main:
            print(
                f"Could not get GPU info for CuPy (is a GPU available and CuPy installed correctly?): {e_gpu_info_main}")
            print("GPU implementation will likely fail if CuPy or GPU is not properly configured.")
    elif not run_gpu:
        print("GPU profiling explicitly disabled for this benchmark run.")

    profile_and_save_stats(
        matrix_dim=size,
        total_runs=runs,
        run_single_thread_impl=run_single_thread,
        run_gpu_impl=run_gpu
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiler for Matrix Multiplication implementations.")
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Dimension of the square matrices (e.g., 1024 for 1024x1024)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=11,
        help="Number of times to run each benchmark."
    )
    parser.add_argument(
        "--no_single_thread",
        action="store_true",
        help="If set, skips the single-threaded CPU implementation."
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="If set, skips the GPU implementation."
    )

    args = parser.parse_args()

    run_single_thread_flag = not args.no_single_thread
    run_gpu_flag = not args.no_gpu

    if args.size >= 2048 and run_single_thread_flag:
        print(f"Note: For matrix dimension {args.size}, single-thread execution might be very slow.")

    run_matrix_multiplication_benchmark(
        size=args.size,
        runs=args.runs,
        run_single_thread=run_single_thread_flag,
        run_gpu=run_gpu_flag
    )
    print("\nMatrix Multiplication profiling complete. Results saved to respective files.")
