import time
import numpy as np
import cupy as cp
from ctypes import cdll, c_void_p, c_int, c_float, POINTER, cast
import os
import traceback

try:
    from sklearn.cluster import KMeans, kmeans_plusplus
    from sklearn.metrics import adjusted_rand_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Verification against scikit-learn will be skipped.")
    print("To enable full verification, please install scikit-learn: `poetry add scikit-learn`")

# --- Configuration ---
dll_filename = "libkmeans_cuda.dll"
cuda_function_name = "kmeans_iteration_gpu"
# --- End Configuration ---

# --- DLL Loading Logic ---
lib = None
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(script_dir, dll_filename)
    if not os.path.exists(dll_path):
        dll_path_cwd = os.path.join(os.getcwd(), dll_filename)
        if os.path.exists(dll_path_cwd):
            dll_path = dll_path_cwd
        else:
            raise FileNotFoundError(f"DLL not found at {dll_path} or {dll_path_cwd}")
    lib = cdll.LoadLibrary(dll_path)
    print(f"Successfully loaded DLL: {dll_path}")
except Exception as e:
    print(f"Error loading DLL '{dll_filename}': {e}")
    lib = None
# --- End DLL Loading ---

if lib:
    try:
        kmeans_iteration_func_dll = getattr(lib, cuda_function_name)
        kmeans_iteration_func_dll.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_int),
            POINTER(c_float), POINTER(c_int),
            c_int, c_int, c_int
        ]
        kmeans_iteration_func_dll.restype = None
        print(f"Successfully found function '{cuda_function_name}' in DLL.")


        def run_kmeans_gpu_custom(
                points_cp: cp.ndarray,
                n_clusters: int,
                initial_centroids_cp: cp.ndarray,
                max_iters: int = 100,
                tol: float = 1e-4):
            """
            Performs K-Means clustering on the GPU using custom CUDA kernels.
            The Python side manages the iteration loop.
            The DLL function performs one iteration (assignment + centroid update).
            """
            N, D = points_cp.shape
            K = n_clusters

            if points_cp.dtype != cp.float32:
                points_cp = points_cp.astype(cp.float32)

            if initial_centroids_cp.shape != (K, D):
                raise ValueError("Initial centroids shape mismatch.")
            if initial_centroids_cp.dtype != cp.float32:
                current_centroids_cp = initial_centroids_cp.astype(cp.float32, copy=True)
            else:
                current_centroids_cp = initial_centroids_cp.copy()

            assignments_cp = cp.empty(N, dtype=cp.int32)
            temp_centroid_sums_cp = cp.zeros((K, D), dtype=cp.float32)
            temp_cluster_counts_cp = cp.zeros(K, dtype=cp.int32)
            old_centroids_cp = cp.empty_like(current_centroids_cp)

            d_points_ptr = cast(points_cp.data.ptr, POINTER(c_float))
            d_centroids_ptr = cast(current_centroids_cp.data.ptr, POINTER(c_float))
            d_assignments_ptr = cast(assignments_cp.data.ptr, POINTER(c_int))
            d_temp_centroid_sums_ptr = cast(temp_centroid_sums_cp.data.ptr, POINTER(c_float))
            d_temp_cluster_counts_ptr = cast(temp_cluster_counts_cp.data.ptr, POINTER(c_int))

            n_iterations_run = 0
            for i in range(max_iters):
                n_iterations_run = i + 1
                cp.copyto(old_centroids_cp, current_centroids_cp)
                kmeans_iteration_func_dll(
                    d_points_ptr, d_centroids_ptr, d_assignments_ptr,
                    d_temp_centroid_sums_ptr, d_temp_cluster_counts_ptr,
                    N, D, K)
                cp.cuda.Stream.null.synchronize()
                diff_sq = cp.sum((current_centroids_cp - old_centroids_cp) ** 2)
                if diff_sq < tol:
                    break
            return current_centroids_cp, assignments_cp, n_iterations_run

    except AttributeError as e_attr:
        print(f"Error: Function '{cuda_function_name}' not found in DLL: {e_attr}")


        def run_kmeans_gpu_custom(*args, **kwargs):
            raise ImportError(f"Function '{cuda_function_name}' not found in DLL.")
else:
    def run_kmeans_gpu_custom(*args, **kwargs):
        raise ImportError(f"DLL '{dll_filename}' could not be loaded.")

if __name__ == "__main__":
    final_centroids_gpu_np = None
    final_assignments_gpu_np = None

    try:
        cp.cuda.Device(0).use()
        print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

        N_points = 100000
        D_dims = 32
        K_clusters = 10
        MAX_ITERS = 50
        TOLERANCE = 1e-4
        RANDOM_STATE = 42

        print(f"\nGenerating {N_points} points, {D_dims} dimensions...")
        points_np = np.random.rand(N_points, D_dims).astype(np.float32)
        points_gpu = cp.asarray(points_np)

        if SKLEARN_AVAILABLE:
            print("Generating initial centroids using k-means++ (sklearn)...")
            initial_centroids_np, _ = kmeans_plusplus(points_np, n_clusters=K_clusters, random_state=RANDOM_STATE)
        else:
            print("Scikit-learn not found. Using random points for initial centroids.")
            indices = np.random.choice(N_points, K_clusters, replace=False)
            initial_centroids_np = points_np[indices].copy()

        initial_centroids_gpu = cp.asarray(initial_centroids_np)
        print("Initial centroids generated and copied to GPU.")

        # --- Run Your Custom GPU K-Means ---
        print(f"\nRunning Custom GPU K-Means with K={K_clusters}, max_iters={MAX_ITERS}...")
        start_time_gpu = time.time()
        final_centroids_gpu, assignments_from_loop_gpu, iters_gpu = run_kmeans_gpu_custom(
            points_gpu, K_clusters, initial_centroids_cp=initial_centroids_gpu.copy(),
            max_iters=MAX_ITERS, tol=TOLERANCE
        )
        cp.cuda.Stream.null.synchronize()
        end_time_gpu = time.time()
        gpu_custom_time = end_time_gpu - start_time_gpu
        print(f"Custom GPU K-Means completed in {iters_gpu} iterations.")
        print(f"Total execution time (Custom GPU): {gpu_custom_time:.4f} seconds.")

        if final_centroids_gpu is not None:
            final_centroids_gpu_np = final_centroids_gpu.get()
        else:
            print("Error: Custom GPU K-Means did not return final centroids.")

        if SKLEARN_AVAILABLE:
            # --- Run scikit-learn K-Means for comparison ---
            print(f"\nRunning scikit-learn K-Means with K={K_clusters}, max_iters={MAX_ITERS} for verification...")
            kmeans_sklearn = KMeans(n_clusters=K_clusters,
                                    init=initial_centroids_np.copy(),
                                    n_init=1,
                                    max_iter=MAX_ITERS,
                                    tol=TOLERANCE,
                                    random_state=RANDOM_STATE,
                                    algorithm='lloyd')
            start_time_sklearn = time.time()
            kmeans_sklearn.fit(points_np)
            end_time_sklearn = time.time()
            sklearn_time = end_time_sklearn - start_time_sklearn
            final_centroids_sklearn_np = kmeans_sklearn.cluster_centers_
            final_assignments_sklearn_np = kmeans_sklearn.labels_
            iters_sklearn = kmeans_sklearn.n_iter_
            inertia_sklearn = kmeans_sklearn.inertia_
            print(f"Scikit-learn K-Means completed in {iters_sklearn} iterations.")
            print(f"Total execution time (scikit-learn): {sklearn_time:.4f} seconds.")

            # --- Verification Results ---
            print("\n--- Verification Results ---")
            print("Performing final assignment pass on GPU with final custom GPU centroids for verification...")
            try:
                dist_sq_all_gpu = cp.sum((points_gpu[:, cp.newaxis, :] - final_centroids_gpu[cp.newaxis, :, :]) ** 2,
                                         axis=2)
                final_assignments_for_verification_cp = cp.argmin(dist_sq_all_gpu, axis=1).astype(cp.int32)
                cp.cuda.Stream.null.synchronize()
                final_assignments_gpu_np = final_assignments_for_verification_cp.get()
            except Exception as e_assign:
                print(f"ERROR during final GPU assignment pass: {e_assign}")
                traceback.print_exc()

            if final_assignments_gpu_np is not None and final_centroids_gpu_np is not None:
                inertia_custom_gpu = 0.0
                for i in range(N_points):
                    cluster_idx = final_assignments_gpu_np[i]
                    if 0 <= cluster_idx < K_clusters:
                        dist_sq = np.sum((points_np[i] - final_centroids_gpu_np[cluster_idx]) ** 2)
                        inertia_custom_gpu += dist_sq
                    else:
                        print(f"Warning: Point {i} (final assignment) has invalid cluster: {cluster_idx}")
                print(f"\nInertia (WCSS):")
                print(f"  Scikit-learn: {inertia_sklearn:.4f}")
                print(f"  Custom GPU:   {inertia_custom_gpu:.4f}")
                print(f"  Difference (sklearn - custom_gpu): {inertia_sklearn - inertia_custom_gpu:.4f}")
                if inertia_sklearn > 1e-9:
                    print(f"  Relative Difference: {abs(inertia_sklearn - inertia_custom_gpu) / inertia_sklearn:.4%}")

                ari_score = adjusted_rand_score(final_assignments_sklearn_np, final_assignments_gpu_np)
                print(f"\nAdjusted Rand Index (similarity of assignments): {ari_score:.4f}")

                if abs(inertia_sklearn - inertia_custom_gpu) / (inertia_sklearn + 1e-9) < 0.01 and ari_score > 0.95:
                    print("Verification NOTE: Results (inertia and assignments) are highly similar to scikit-learn.")
                else:
                    print(
                        "Verification NOTE: Results show differences. Could be convergence to different local optima or implementation nuances.")
            else:
                print("Skipping Scikit-learn comparison metrics due to missing GPU assignments or centroids.")

        else:
            if final_centroids_gpu is not None:
                print(
                    "Performing final assignment pass on GPU with final custom GPU centroids for self-consistency (sklearn not available)...")
                try:
                    dist_sq_all_gpu = cp.sum(
                        (points_gpu[:, cp.newaxis, :] - final_centroids_gpu[cp.newaxis, :, :]) ** 2, axis=2)
                    final_assignments_for_verification_cp = cp.argmin(dist_sq_all_gpu, axis=1).astype(cp.int32)
                    cp.cuda.Stream.null.synchronize()
                    final_assignments_gpu_np = final_assignments_for_verification_cp.get()
                except Exception as e_assign_no_sklearn:
                    print(f"ERROR during final GPU assignment pass (no sklearn): {e_assign_no_sklearn}")
                    traceback.print_exc()
            else:
                print("Skipping final GPU assignment pass because final_centroids_gpu is None.")

        # --- Self-Consistency Check for your GPU implementation ---
        if final_assignments_gpu_np is not None and final_centroids_gpu_np is not None:
            print("\n--- Self-Consistency Check (Custom GPU results with final assignments) ---")
            correct_assignments_self = 0
            for i in range(N_points):
                point = points_np[i]
                assigned_cluster = final_assignments_gpu_np[i]

                min_dist_sq_to_final_centroids = np.inf
                closest_cluster_idx = -1
                for k_idx in range(K_clusters):
                    dist_sq = np.sum((point - final_centroids_gpu_np[k_idx]) ** 2)
                    if dist_sq < min_dist_sq_to_final_centroids:
                        min_dist_sq_to_final_centroids = dist_sq
                        closest_cluster_idx = k_idx

                if assigned_cluster == closest_cluster_idx:
                    correct_assignments_self += 1

            assignment_accuracy_self = (correct_assignments_self / N_points) * 100 if N_points > 0 else 0
            print(
                f"  Assignment Optimality: {assignment_accuracy_self:.2f}% of points are assigned to their closest final GPU centroid.")
            if assignment_accuracy_self < 99.99:
                print(
                    "  Warning: Self-consistency check for assignments is not 100%. Ideally, all points should be closest to their assigned final centroid.")
        else:
            print(
                "\nSelf-Consistency Check SKIPPED as final_assignments_gpu_np or final_centroids_gpu_np was not successfully defined/retrieved.")

    except ImportError as e_main:
        if 'sklearn' in str(e_main).lower() and not SKLEARN_AVAILABLE:
            pass
        else:
            print(f"ImportError: {e_main}")
    except cp.cuda.memory.OutOfMemoryError:
        print("\nCUDA Out of Memory Error: Array size might be too large for GPU memory.")
        traceback.print_exc()
    except Exception as e_main:
        print(f"An error occurred during k-means execution: {e_main}")
        traceback.print_exc()

    # --- Test complete ---
    print(f"\n--- Test complete ---")