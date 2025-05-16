import numpy as np
import time


def calculate_squared_distances_st(points, centroids):
    """
    Calculates squared Euclidean distances from each point to each centroid.
    points: (N, D) array of N points, D dimensions
    centroids: (K, D) array of K centroids, D dimensions
    Returns: (N, K) array of squared distances
    """
    distances_sq = np.sum((points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    return distances_sq


def assign_points_st(points, centroids):
    """
    Assigns each point to the nearest centroid.
    Returns: (N,) array of cluster assignments (indices 0 to K-1)
    """
    distances_sq = calculate_squared_distances_st(points, centroids)
    assignments = np.argmin(distances_sq, axis=1)
    return assignments.astype(np.int32)


def update_centroids_st(points, assignments, n_clusters, n_features):
    """
    Recalculates centroids as the mean of assigned points.
    Returns: (K, D) array of new centroids
    """
    new_centroids = np.zeros((n_clusters, n_features), dtype=points.dtype)
    cluster_counts = np.zeros(n_clusters, dtype=np.intp)

    for i in range(points.shape[0]):
        cluster_idx = assignments[i]
        if 0 <= cluster_idx < n_clusters:
            new_centroids[cluster_idx] += points[i]
            cluster_counts[cluster_idx] += 1

    for k in range(n_clusters):
        if cluster_counts[k] > 0:
            new_centroids[k] /= cluster_counts[k]

    return new_centroids


def kmeans_single_thread(points: np.ndarray,
                         n_clusters: int,
                         initial_centroids: np.ndarray = None,
                         max_iters: int = 100,
                         tol: float = 1e-4):
    """
    Single-threaded k-Means clustering using NumPy.
    """
    n_samples, n_features = points.shape

    if points.dtype != np.float32:
        points_internal = points.astype(np.float32, copy=True)
    else:
        points_internal = points

    if initial_centroids is None:
        random_indices = np.random.choice(n_samples, n_clusters, replace=False)
        current_centroids = points_internal[random_indices].copy()
    else:
        if initial_centroids.shape != (n_clusters, n_features):
            raise ValueError("Initial centroids shape mismatch.")
        if initial_centroids.dtype != np.float32:
            current_centroids = initial_centroids.astype(np.float32, copy=True)
        else:
            current_centroids = initial_centroids.copy()

    n_iterations_run = 0
    for i in range(max_iters):
        n_iterations_run = i + 1
        old_centroids = current_centroids.copy()

        assignments = assign_points_st(points_internal, current_centroids)
        current_centroids = update_centroids_st(points_internal, assignments, n_clusters, n_features)

        centroid_shift_sq = np.sum((current_centroids - old_centroids) ** 2)
        if centroid_shift_sq < tol:
            break

    return current_centroids, assignments, n_iterations_run


if __name__ == '__main__':
    N_points = 100000
    D_dims = 32
    K_clusters = 10
    MAX_ITERS_TEST = 50
    TOLERANCE_TEST = 1e-4
    RANDOM_STATE_TEST = 42
    np.random.seed(RANDOM_STATE_TEST)

    print(f"Generating {N_points} points, {D_dims}D for K-Means single-thread test...")
    test_points_np = np.random.rand(N_points, D_dims).astype(np.float32)

    try:
        from sklearn.cluster import kmeans_plusplus

        initial_centroids_for_test, _ = kmeans_plusplus(test_points_np, n_clusters=K_clusters,
                                                        random_state=RANDOM_STATE_TEST)
        initial_centroids_for_test = initial_centroids_for_test.astype(np.float32)
    except ImportError:
        print("sklearn not available for kmeans_plusplus init, using random subset.")
        random_indices = np.random.choice(N_points, K_clusters, replace=False)
        initial_centroids_for_test = test_points_np[random_indices].copy()

    print(f"Running single-threaded K-Means for {N_points} points, K={K_clusters}...")

    start_t = time.time()
    final_centroids_st, final_assignments_st, iters_st = kmeans_single_thread(
        test_points_np,
        K_clusters,
        initial_centroids=initial_centroids_for_test,
        max_iters=MAX_ITERS_TEST,
        tol=TOLERANCE_TEST
    )
    end_t = time.time()

    print(f"Single-threaded K-Means completed in {iters_st} iterations.")
    print(f"Execution time: {end_t - start_t:.4f} seconds.")