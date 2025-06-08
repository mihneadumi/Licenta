import os
import time
import numpy as np
import cupy as cp
import traceback
import platform
import psutil
import argparse

from algorithms.radix_sort.optimised_radix_sort import radix_sort as radix_sort_single_thread_optimised
from algorithms.radix_sort.multi_threaded import radix_sort_numba
from algorithms.radix_sort.gpu_acceleration import radix_sort_gpu
from utils.utils import get_gpu_info

RESULTS_BASE_PATH = 'results/'
RADIX_SORT_PATH = 'radix_sort/'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RANDOM_SEED = 42
DATA_TYPE_RADIX = np.uint32
UPPER_BOUND_RADIX = 2 ** 32


def generate_array(array_size, random_state_seed, dtype=DATA_TYPE_RADIX, upper_bound=UPPER_BOUND_RADIX):
    rng = np.random.RandomState(random_state_seed)
    arr_np = rng.randint(0, upper_bound, size=array_size, dtype=dtype)
    return arr_np


def profile_and_save_stats(
        array_size: int,
        total_runs: int,
        run_single_thread_impl: bool = True,
        run_gpu_impl: bool = True
):
    size_str = f"S{array_size}"
    print(f"\nInfo: Profiling Radix Sort for configuration: {size_str} (Size: {array_size:,})")
    print(f"Parameters: Runs={total_runs}, Data Type={DATA_TYPE_RADIX.__name__}")
    if not run_single_thread_impl:
        print("  NOTE: Single-threaded CPU (Optimised) implementation will be SKIPPED.")
    if not run_gpu_impl:
        print("  NOTE: GPU implementation will be SKIPPED.")

    output_dir = os.path.join(RESULTS_BASE_PATH, RADIX_SORT_PATH, str(array_size))
    os.makedirs(output_dir, exist_ok=True)

    impl_config = {
        "single_thread_optimised": {
            "file_suffix": 'optimised_cpu_single_thread_stats.txt',
            "func": radix_sort_single_thread_optimised,
            "run_this_time": run_single_thread_impl,
            "is_gpu": False,
            "name_print": "CPU Single-Thread Radix (Optimised)",
            "returns_new_array": False
        },
        "multi_thread_numba": {
            "file_suffix": 'cpu_numba_stats.txt',
            "func": radix_sort_numba,
            "run_this_time": True,
            "is_gpu": False,
            "name_print": "CPU Multi-Thread Radix (Numba)",
            "returns_new_array": True
        },
        "gpu_custom": {
            "file_suffix": 'gpu_acceleration_stats.txt',
            "func": radix_sort_gpu,
            "run_this_time": run_gpu_impl,
            "is_gpu": True,
            "name_print": "GPU Custom Radix",
            "returns_new_array": True
        }
    }

    file_handles = {}
    active_implementations_for_run = {}

    try:
        for key, config_item in impl_config.items():
            if config_item["run_this_time"]:
                path = os.path.join(output_dir, config_item["file_suffix"])
                file_handles[key] = open(path, 'w')
                file_handles[key].write("Run,Timestamp,Time(s),Size,MElementsPerSec\n")  # Changed ArraySize to Size
                active_implementations_for_run[key] = config_item

        if not active_implementations_for_run:
            print(f"    No Radix Sort implementations selected to run for {size_str}. Skipping.")
            return

        if "multi_thread_numba" in active_implementations_for_run:
            print("  Warming up Numba Radix Sort JIT compiler...")
            dummy_array_np_warmup = generate_array(min(1000, array_size), RANDOM_SEED - 1)
            try:
                _ = radix_sort_numba(dummy_array_np_warmup.copy())
                print("  Numba Radix Sort warm-up complete.")
            except Exception as e_warmup:
                print(f"  Warning: Numba Radix Sort warm-up failed: {e_warmup}")

        for run_number in range(1, total_runs + 1):
            print(f"  Starting Run {run_number}/{total_runs} for {size_str}...")
            current_run_seed = RANDOM_SEED + run_number

            arr_np_original = generate_array(array_size, current_run_seed)

            for impl_key, config_item in active_implementations_for_run.items():
                func_to_profile = config_item["func"]
                impl_name_print = config_item["name_print"]
                is_gpu = config_item["is_gpu"]
                returns_new = config_item["returns_new_array"]

                if is_gpu:
                    arr_input = cp.asarray(arr_np_original.copy())
                elif returns_new:
                    arr_input = arr_np_original.copy()
                else:
                    arr_input = arr_np_original.copy()

                print(f"    Profiling {impl_name_print}...")
                exec_time = float('inf')
                melements_per_sec = 0.0

                try:
                    if is_gpu:
                        cp.cuda.Stream.null.synchronize()

                    start_time = time.perf_counter()
                    _ = func_to_profile(arr_input)

                    if is_gpu:
                        cp.cuda.Stream.null.synchronize()
                    end_time = time.perf_counter()
                    exec_time = end_time - start_time

                    if exec_time > 0:
                        elements_per_sec = array_size / exec_time
                        melements_per_sec = elements_per_sec / 1e6

                    timestamp = time.strftime(DATE_FORMAT)
                    # array_size variable corresponds to the "Size" column
                    result_line = (f"{run_number},{timestamp},{exec_time:.6f},"
                                   f"{array_size},{melements_per_sec:.2f}\n")
                    file_handles[impl_key].write(result_line)
                    print(
                        f"      {impl_name_print} Run {run_number}: {exec_time:.6f}s, "
                        f"MElements/s: {melements_per_sec:.2f}")

                except Exception as e:
                    print(f"      Error during {impl_name_print} profiling for run {run_number}: {e}")
                    traceback.print_exc()
                    timestamp = time.strftime(DATE_FORMAT)
                    # array_size variable corresponds to the "Size" column
                    result_line = (f"{run_number},{timestamp},inf,{array_size},0.0\n")
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


def run_radix_sort_benchmark(size: int, runs: int, run_single_thread: bool = True, run_gpu: bool = True):
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
            print(f"Could not get GPU info for CuPy: {e_gpu_info_main}")
            print("GPU implementation will likely fail if CuPy or GPU is not properly configured.")
    elif not run_gpu:
        print("GPU profiling explicitly disabled for this Radix Sort benchmark run.")

    profile_and_save_stats(
        array_size=size,
        total_runs=runs,
        run_single_thread_impl=run_single_thread,
        run_gpu_impl=run_gpu
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiler for Radix Sort implementations.")
    parser.add_argument(
        "--size",
        type=int,
        default=1000000,
        help="Number of elements in the array to sort."
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
        help="If set, skips the single-threaded CPU (Optimised) implementation."
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="If set, skips the GPU implementation."
    )

    args = parser.parse_args()

    run_single_thread_flag = not args.no_single_thread
    run_gpu_flag = not args.no_gpu

    if args.size >= 10000000 and run_single_thread_flag:
        print(f"Note: For array size {args.size:,}, single-thread Radix Sort execution might be very slow.")

    run_radix_sort_benchmark(
        size=args.size,
        runs=args.runs,
        run_single_thread=run_single_thread_flag,
        run_gpu=run_gpu_flag
    )
    print("\nRadix Sort profiling complete. Results saved to respective files.")
