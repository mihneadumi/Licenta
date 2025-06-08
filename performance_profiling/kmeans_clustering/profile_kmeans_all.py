import os
import time
import numpy as np
import cupy as cp
import traceback
import platform
import psutil

# --- Attempt to import scikit-learn for consistent centroid initialization ---
try:
    from sklearn.cluster import kmeans_plusplus

    SKLEARN_AVAILABLE_FOR_INIT = True
except ImportError:
    SKLEARN_AVAILABLE_FOR_INIT = False
    print("Warning: scikit-learn (for kmeans_plusplus) not found. Initial centroids will be random samples.")

# --- Attempt to import the k-Means implementations ---
try:
    from algorithms.kmeans_clustering.single_thread import kmeans_single_thread

    SINGLE_THREAD_KMEANS_AVAILABLE = True
except ImportError:
    SINGLE_THREAD_KMEANS_AVAILABLE = False
    print("Error: Could not import 'kmeans_single_thread'. Using dummy if called.")


    def kmeans_single_thread(points, n_clusters, initial_centroids=None, max_iters=100, tol=1e-4):
        print("Warning: Using dummy kmeans_single_thread.")
        time.sleep(0.1)
        return initial_centroids if initial_centroids is not None else np.zeros((n_clusters, points.shape[1]),
                                                                                dtype=points.dtype), \
            np.zeros(points.shape[0], dtype=np.int32), max_iters

try:
    from algorithms.kmeans_clustering.multi_thread import kmeans_parallel_numba

    NUMBA_KMEANS_AVAILABLE = True
except ImportError:
    NUMBA_KMEANS_AVAILABLE = False
    print("Error: Could not import 'kmeans_parallel_numba'. Using dummy if called.")


    def kmeans_parallel_numba(points, n_clusters, initial_centroids=None, max_iters=100, tol=1e-4):
        print("Warning: Using dummy kmeans_parallel_numba.")
        time.sleep(0.05)
        return initial_centroids if initial_centroids is not None else np.zeros((n_clusters, points.shape[1]),
                                                                                dtype=points.dtype), \
            np.zeros(points.shape[0], dtype=np.int32), max_iters

try:
    from algorithms.kmeans_clustering.gpu_acceleration import run_kmeans_gpu_custom

    GPU_KMEANS_AVAILABLE = True
except ImportError:
    GPU_KMEANS_AVAILABLE = False
    print("Error: Could not import 'run_kmeans_gpu_custom'. Using dummy if called.")


    def run_kmeans_gpu_custom(points_cp, n_clusters, initial_centroids_cp=None, max_iters=100, tol=1e-4):
        print("Warning: Using dummy run_kmeans_gpu_custom.")
        cp.cuda.Stream.null.synchronize()
        time.sleep(0.01)
        return initial_centroids_cp if initial_centroids_cp is not None else cp.zeros((n_clusters, points_cp.shape[1]),
                                                                                      dtype=points_cp.dtype), \
            cp.zeros(points_cp.shape[0], dtype=cp.int32), max_iters

# --- Configuration ---
RESULTS_BASE_PATH = 'results/'
KMEANS_CLUSTERING_PATH = 'kmeans_clustering/'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RANDOM_SEED = 42


# --- Helper Functions ---
def generate_data_and_initial_centroids(n_points, n_dims, n_clusters, random_state_seed):
    """Generates random data and initial centroids."""
    points_np = np.random.rand(n_points, n_dims).astype(np.float32)
    initial_centroids_np = None
    if SKLEARN_AVAILABLE_FOR_INIT:
        try:
            initial_centroids_np, _ = kmeans_plusplus(points_np, n_clusters=n_clusters, random_state=random_state_seed)
            initial_centroids_np = initial_centroids_np.astype(np.float32)
        except Exception as e_kmpp:
            print(f"    Warning: k-means++ initialization failed ({e_kmpp}). Falling back to random sampling.")
            rng = np.random.RandomState(random_state_seed)
            indices = rng.choice(n_points, n_clusters, replace=False)
            initial_centroids_np = points_np[indices].copy()
    else:
        rng = np.random.RandomState(random_state_seed)
        indices = rng.choice(n_points, n_clusters, replace=False)
        initial_centroids_np = points_np[indices].copy()
    return points_np, initial_centroids_np


def profile_and_save_stats(
        n_points: int, n_dims: int, n_clusters: int,
        max_iters: int, tol: float, total_runs: int,
        run_single_thread_impl: bool = True
):
    """
    Profiles k-Means implementations for given parameters and saves statistics.
    Can skip single-threaded implementation if run_single_thread_impl is False.
    """
    size_str = f"N{n_points}_D{n_dims}_K{n_clusters}"
    print(f"\nInfo: Profiling K-Means for configuration: {size_str}")
    print(f"Parameters: Max Iterations={max_iters}, Tolerance={tol}, Runs={total_runs}")
    if not run_single_thread_impl:
        print("  NOTE: Single-threaded CPU implementation will be SKIPPED for this configuration.")

    output_dir = os.path.join(RESULTS_BASE_PATH, KMEANS_CLUSTERING_PATH, size_str)
    os.makedirs(output_dir, exist_ok=True)

    impl_config = {
        "single_thread": {
            "file_suffix": 'cpu_single_thread_stats.txt',
            "func": kmeans_single_thread,
            "available": SINGLE_THREAD_KMEANS_AVAILABLE,
            "run_this_time": run_single_thread_impl and SINGLE_THREAD_KMEANS_AVAILABLE,
            "is_gpu": False,
            "name_print": "CPU Single-Thread"
        },
        "parallel_numba": {
            "file_suffix": 'cpu_parallel_numba_stats.txt',
            "func": kmeans_parallel_numba,
            "available": NUMBA_KMEANS_AVAILABLE,
            "run_this_time": NUMBA_KMEANS_AVAILABLE,
            "is_gpu": False,
            "name_print": "CPU Parallel Numba"
        },
        "gpu_custom": {
            "file_suffix": 'gpu_acceleration_stats.txt',
            "func": run_kmeans_gpu_custom,
            "available": GPU_KMEANS_AVAILABLE,
            "run_this_time": GPU_KMEANS_AVAILABLE,
            "is_gpu": True,
            "name_print": "GPU Custom CUDA"
        }
    }

    file_handles = {}
    active_implementations_for_run = {}

    try:
        for key, config_item in impl_config.items():
            if config_item["run_this_time"]:
                path = os.path.join(output_dir, config_item["file_suffix"])
                file_handles[key] = open(path, 'w')
                file_handles[key].write("Run,Timestamp,Time(s),N_Points,D_Dims,K_Clusters,IterationsRun,PointsPerSec\n")
                active_implementations_for_run[key] = config_item

        if not active_implementations_for_run:
            print(f"    No k-Means implementations available or selected to run for {size_str}. Skipping.")
            return

        if NUMBA_KMEANS_AVAILABLE and "parallel_numba" in active_implementations_for_run:
            print("  Warming up Parallel Numba K-Means JIT compiler...")
            dummy_points_np_warmup, dummy_initial_centroids_np_warmup = generate_data_and_initial_centroids(100, n_dims,
                                                                                                            n_clusters,
                                                                                                            RANDOM_SEED)
            if dummy_initial_centroids_np_warmup is not None:
                try:
                    kmeans_parallel_numba(dummy_points_np_warmup, n_clusters,
                                          initial_centroids=dummy_initial_centroids_np_warmup, max_iters=2, tol=tol)
                    print("  Numba warm-up complete.")
                except Exception as e_warmup:
                    print(f"  Warning: Numba warm-up failed: {e_warmup}")
            else:
                print("  Skipping Numba warm-up due to centroid generation issue.")

        for run_number in range(1, total_runs + 1):
            print(f"  Starting Run {run_number}/{total_runs} for {size_str}...")
            current_run_seed = RANDOM_SEED + run_number
            points_np, initial_centroids_np = generate_data_and_initial_centroids(
                n_points, n_dims, n_clusters, current_run_seed
            )
            if initial_centroids_np is None:
                print(f"    FATAL: Could not generate initial centroids for run {run_number}. Skipping run.")
                continue

            for impl_key, config_item in active_implementations_for_run.items():
                func = config_item["func"]
                impl_name_print = config_item["name_print"]
                is_gpu = config_item["is_gpu"]

                data_input = cp.asarray(points_np) if is_gpu else points_np.copy()
                init_centroids_input = cp.asarray(initial_centroids_np) if is_gpu else initial_centroids_np.copy()

                print(f"    Profiling {impl_name_print}...")
                exec_time = float('inf')
                iters_run = 0
                try:
                    start_time = time.time()
                    if is_gpu:
                        _, _, iters_run = func(
                            data_input, n_clusters, initial_centroids_cp=init_centroids_input.copy(),
                            max_iters=max_iters, tol=tol
                        )
                        cp.cuda.Stream.null.synchronize()
                    else:
                        _, _, iters_run = func(
                            data_input, n_clusters, initial_centroids=init_centroids_input.copy(),
                            max_iters=max_iters, tol=tol
                        )
                    end_time = time.time()
                    exec_time = end_time - start_time

                    points_per_sec = n_points / exec_time if exec_time > 0 else 0.0
                    timestamp = time.strftime(DATE_FORMAT)
                    result_line = f"{run_number},{timestamp},{exec_time:.4f},{n_points},{n_dims},{n_clusters},{iters_run},{points_per_sec:.2f}\n"
                    file_handles[impl_key].write(result_line)
                    print(
                        f"      {impl_name_print} Run {run_number}: {exec_time:.4f}s, Iterations: {iters_run}, Throughput: {points_per_sec:.2f} Points/s")
                except Exception as e:
                    print(f"      Error during {impl_name_print} profiling for run {run_number}: {e}")
                    traceback.print_exc()
                    timestamp = time.strftime(DATE_FORMAT)
                    result_line = f"{run_number},{timestamp},inf,{n_points},{n_dims},{n_clusters},0,0.0\n"
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


def run_all_kmeans_benchmarks(include_single_thread_for_standard_tests: bool = True):
    """Defines and runs a suite of k-Means benchmarks."""
    benchmark_params = [
        (100000, 32, 10, 50, 1e-4, 11),
        (1000000, 32, 10, 50, 1e-4, 11),
        (10000000, 32, 10, 50, 1e-4, 11),
    ]
    try:
        print(f"CPU Info: {platform.processor()}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    except Exception as e_cpu_info:
        print(f"Could not get CPU info: {e_cpu_info}")

    if GPU_KMEANS_AVAILABLE:
        try:
            cp.cuda.Device(0).use()
            from utils.utils import get_gpu_info
            print(f"GPU Info: {get_gpu_info()}")
        except ImportError:
            print("Warning: utils.utils.get_gpu_info not found. Using basic CuPy name.")
            try:
                print(f"GPU Info: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            except Exception as e_gpu_info_fallback:
                print(f"Could not get GPU name via CuPy: {e_gpu_info_fallback}")
        except Exception as e_gpu_info_main:
            print(f"Could not get GPU info for CuPy: {e_gpu_info_main}")

    for params_tuple in benchmark_params:
        profile_and_save_stats(*params_tuple, run_single_thread_impl=include_single_thread_for_standard_tests)


if __name__ == "__main__":
    run_all_kmeans_benchmarks()
    print("\nK-Means profiling complete. Results saved to respective files.")