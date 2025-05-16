import numpy as np
import numba
from numba import njit, prange
import time

if not hasattr(numba, 'get_num_threads'):
    try:
        from numba.np.ufunc.parallel import get_num_threads as numba_get_num_threads
        from numba.core.config import NUMBA_DEFAULT_NUM_THREADS
        from numba.np.ufunc.parallel import get_thread_id as numba_get_thread_id

        numba.get_num_threads = numba_get_num_threads
        numba.get_thread_id = numba_get_thread_id
    except ImportError:
        numba.get_num_threads = lambda: getattr(numba.core.config, 'NUMBA_DEFAULT_NUM_THREADS', 1)
        numba.get_thread_id = lambda: 0


@njit(parallel=True, cache=True)
def assign_points_numba_parallel(points, centroids, assignments, N, D, K):
    """Assigns each point to the nearest centroid (parallelized over points)."""
    for i in prange(N):
        min_dist_sq = np.inf
        best_cluster_idx = -1
        current_point_i_data = points[i]

        for k in range(K):
            dist_sq = 0.0
            current_centroid_k_data = centroids[k]
            for dim in range(D):
                diff = current_point_i_data[dim] - current_centroid_k_data[dim]
                dist_sq += diff * diff

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_cluster_idx = k
        assignments[i] = best_cluster_idx


@njit(parallel=True, cache=True)
def update_centroids_numba_parallel(points, assignments,
                                    new_centroids_out, cluster_counts_out,
                                    N, D, K):
    """
    Recalculates centroids using per-thread accumulators, then a serial reduction.
    new_centroids_out and cluster_counts_out are modified in-place.
    """
    num_threads = numba.get_num_threads()

    per_thread_centroid_sums = np.zeros((num_threads, K, D), dtype=points.dtype)
    per_thread_cluster_counts = np.zeros((num_threads, K), dtype=np.intp)

    for i in prange(N):
        thread_id = numba.get_thread_id()
        cluster_idx = assignments[i]
        if 0 <= cluster_idx < K:
            for dim in range(D):
                per_thread_centroid_sums[thread_id, cluster_idx, dim] += points[i, dim]
            per_thread_cluster_counts[thread_id, cluster_idx] += 1

    for k_idx in range(K):
        cluster_counts_out[k_idx] = 0
        for dim_idx in range(D):
            new_centroids_out[k_idx, dim_idx] = 0.0

    for t in range(num_threads):
        for k_idx in range(K):
            cluster_counts_out[k_idx] += per_thread_cluster_counts[t, k_idx]
            for dim_idx in range(D):
                new_centroids_out[k_idx, dim_idx] += per_thread_centroid_sums[t, k_idx, dim_idx]

    for k_idx in range(K):
        if cluster_counts_out[k_idx] > 0:
            for dim_idx in range(D):
                new_centroids_out[k_idx, dim_idx] /= cluster_counts_out[k_idx]


def kmeans_parallel_numba(points: np.ndarray,
                          n_clusters: int,
                          initial_centroids: np.ndarray = None,
                          max_iters: int = 100,
                          tol: float = 1e-4):
    """
    Parallel K-Means clustering using Numba.
    """
    n_samples, n_features = points.shape
    K = n_clusters

    if points.dtype != np.float32:
        points_internal = points.astype(np.float32, copy=True)
    else:
        points_internal = points

    if initial_centroids is None:
        random_indices = np.random.choice(n_samples, K, replace=False)
        current_centroids = points_internal[random_indices].copy()
    else:
        if initial_centroids.shape != (K, n_features):
            raise ValueError("Initial centroids shape mismatch.")
        if initial_centroids.dtype != np.float32:
            current_centroids = initial_centroids.astype(np.float32, copy=True)
        else:
            current_centroids = initial_centroids.copy()

    assignments = np.empty(n_samples, dtype=np.int32)
    new_centroids_buffer = np.zeros_like(current_centroids)
    cluster_counts_buffer = np.zeros(K, dtype=np.intp)

    n_iterations_run = 0
    for i in range(max_iters):
        n_iterations_run = i + 1
        old_centroids = current_centroids.copy()

        assign_points_numba_parallel(points_internal, current_centroids, assignments, n_samples, n_features, K)

        update_centroids_numba_parallel(points_internal, assignments,
                                        new_centroids_buffer, cluster_counts_buffer,
                                        n_samples, n_features, K)
        current_centroids = new_centroids_buffer.copy()

        centroid_shift_sq = np.sum((current_centroids - old_centroids) ** 2)
        if centroid_shift_sq < tol:
            break

    assign_points_numba_parallel(points_internal, current_centroids, assignments, n_samples, n_features, K)

    return current_centroids, assignments, n_iterations_run


if __name__ == '__main__':
    N_points = 100000
    D_dims = 32
    K_clusters = 10
    MAX_ITERS_TEST = 50
    TOLERANCE_TEST = 1e-4
    RANDOM_STATE_TEST = 42
    np.random.seed(RANDOM_STATE_TEST)

    print(f"Generating {N_points} points, {D_dims}D for Parallel Numba K-Means test...")
    test_points_np = np.random.rand(N_points, D_dims).astype(np.float32)

    try:
        from sklearn.cluster import kmeans_plusplus

        initial_centroids_for_test, _ = kmeans_plusplus(test_points_np, n_clusters=K_clusters,
                                                        random_state=RANDOM_STATE_TEST)
        initial_centroids_for_test = initial_centroids_for_test.astype(np.float32)
    except ImportError:
        print("sklearn not available for kmeans_plusplus init, using random subset for Numba test.")
        random_indices = np.random.choice(N_points, K_clusters, replace=False)
        initial_centroids_for_test = test_points_np[random_indices].copy()

    print("Performing Numba JIT compilation run (warm-up)...")
    _, _, _ = kmeans_parallel_numba(test_points_np[:100], K_clusters,
                                    initial_centroids=initial_centroids_for_test[:K_clusters, :D_dims].copy() if
                                    initial_centroids_for_test.shape[0] >= K_clusters else None, max_iters=2)

    print(f"\nRunning Parallel Numba K-Means for {N_points} points, K={K_clusters}...")
    start_t = time.time()
    final_centroids_mt, final_assignments_mt, iters_mt = kmeans_parallel_numba(
        test_points_np,
        K_clusters,
        initial_centroids=initial_centroids_for_test,
        max_iters=MAX_ITERS_TEST,
        tol=TOLERANCE_TEST
    )
    end_t = time.time()

    print(f"Parallel Numba K-Means completed in {iters_mt} iterations.")
    print(f"Execution time: {end_t - start_t:.4f} seconds.")