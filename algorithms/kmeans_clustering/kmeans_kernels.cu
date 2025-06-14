#include <cuda_runtime.h>
#include <float.h> // For FLT_MAX
#include <stdio.h>
#include <stdint.h>

#define DLL_EXPORT __declspec(dllexport)

// --- Helper Macros for Error Checking (same as your Radix Sort) ---
#define CHECK_CUDA_API(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA API Error at %s (%s:%d): %s\n", msg, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

#define CHECK_KERNEL_LAUNCH(msg) \
    do { \
        cudaError_t err = cudaPeekAtLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Launch Error before %s (%s:%d): %s\n", msg, __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error after %s kernel sync (%s:%d): %s\n", msg, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

// --- Kernel for Assigning Points to Nearest Centroids ---
// Each thread processes one data point.
__global__ void assign_points_to_centroids_kernel(
    const float* __restrict__ points,      // Input: Data points (N x D)
    const float* __restrict__ centroids,   // Input: Current centroids (K x D)
    int* __restrict__ assignments,         // Output: Cluster assignment for each point (N)
    int N,                                 // Number of points
    int D,                                 // Number of dimensions
    int K)                                 // Number of clusters
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < N) {
        float min_dist_sq = FLT_MAX;
        int best_cluster_idx = -1;

        // Iterate over each centroid
        for (int k = 0; k < K; ++k) {
            float current_dist_sq = 0.0f;
            // Calculate squared Euclidean distance
            for (int dim = 0; dim < D; ++dim) {
                // points[point_idx * D + dim] accesses the 'dim'-th dimension of 'point_idx'
                // centroids[k * D + dim] accesses the 'dim'-th dimension of centroid 'k'
                float diff = points[point_idx * D + dim] - centroids[k * D + dim];
                current_dist_sq += diff * diff;
            }

            if (current_dist_sq < min_dist_sq) {
                min_dist_sq = current_dist_sq;
                best_cluster_idx = k;
            }
        }
        assignments[point_idx] = best_cluster_idx;
    }
}

// --- Kernel(s) for Updating Centroids ---
// Implements centroid recalculation via parallel accumulation using global atomics.
// This approach is straightforward but can be slower than more advanced parallel
// reduction techniques, especially for high-dimensional data but it's the best I got.

// Kernel to initialize sums and counts for new centroids
__global__ void initialize_centroid_accumulators_kernel(
    float* new_centroid_sums, // K x D, zeroed out
    int* cluster_counts,      // K, zeroed out
    int K, int D)
{
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_idx < K) {
        cluster_counts[cluster_idx] = 0;
        for (int d = 0; d < D; ++d) {
            new_centroid_sums[cluster_idx * D + d] = 0.0f;
        }
    }
}

// Kernel to accumulate point coordinates and counts for each cluster using atomics
// This is a basic approach; more optimized reductions are possible.
__global__ void accumulate_points_for_centroids_kernel(
    const float* __restrict__ points,          // Input: Data points (N x D)
    const int* __restrict__ assignments,       // Input: Cluster assignment for each point (N)
    float* __restrict__ new_centroid_sums,     // Output: Sum of points per cluster (K x D) - updated atomically
    int* __restrict__ cluster_counts,          // Output: Count of points per cluster (K) - updated atomically
    int N, int D, int K)
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < N) {
        int cluster_idx = assignments[point_idx];
        if (cluster_idx >= 0 && cluster_idx < K) { // Basic check
            atomicAdd(&cluster_counts[cluster_idx], 1);
            for (int d = 0; d < D; ++d) {
                atomicAdd(&new_centroid_sums[cluster_idx * D + d], points[point_idx * D + d]);
            }
        }
    }
}

// Kernel to finalize new centroids (divide sums by counts)
__global__ void finalize_centroids_kernel(
    const float* __restrict__ new_centroid_sums, // Input: Sum of points per cluster (K x D)
    const int* __restrict__ cluster_counts,      // Input: Count of points per cluster (K)
    float* __restrict__ new_centroids,           // Output: New centroids (K x D)
    int K, int D)
{
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx < K) {
        int count = cluster_counts[cluster_idx];
        if (count > 0) {
            for (int d = 0; d < D; ++d) {
                new_centroids[cluster_idx * D + d] = new_centroid_sums[cluster_idx * D + d] / (float)count;
            }
        }
    }
}


// --- Host Function (DLL Export) for one K-Means Iteration ---
extern "C" DLL_EXPORT void kmeans_iteration_gpu(
    const float* d_points,      // N x D
    float* d_centroids,         // K x D (input current, output new) - will be updated in-place
    int* d_assignments,         // N (output from assignment step)
    float* d_temp_centroid_sums,// K x D (temporary buffer for sums)
    int* d_temp_cluster_counts, // K (temporary buffer for counts)
    int N, int D, int K)
{
    // --- 1. Assign points to current centroids ---
    dim3 assign_block_dim(256);
    dim3 assign_grid_dim((N + assign_block_dim.x - 1) / assign_block_dim.x);
    assign_points_to_centroids_kernel<<<assign_grid_dim, assign_block_dim>>>(
        d_points, d_centroids, d_assignments, N, D, K);
    CHECK_KERNEL_LAUNCH("assign_points_to_centroids_kernel");

    // --- 2. Update centroids ---
    // This involves:
    // a. Initialize/zero out accumulators for sums and counts
    // b. Accumulate sums and counts based on new assignments
    // c. Finalize new centroids by dividing sums by counts

    dim3 init_finalize_block_dim(256); // Can use same block size
    dim3 init_finalize_grid_dim((K + init_finalize_block_dim.x - 1) / init_finalize_block_dim.x);

    // a. Initialize temporary accumulators
    initialize_centroid_accumulators_kernel<<<init_finalize_grid_dim, init_finalize_block_dim>>>(
        d_temp_centroid_sums, d_temp_cluster_counts, K, D);
    CHECK_KERNEL_LAUNCH("initialize_centroid_accumulators_kernel");

    // b. Accumulate sums and counts
    // Grid dim for accumulation should be based on N (number of points)
    accumulate_points_for_centroids_kernel<<<assign_grid_dim, assign_block_dim>>>(
        d_points, d_assignments, d_temp_centroid_sums, d_temp_cluster_counts, N, D, K);
    CHECK_KERNEL_LAUNCH("accumulate_points_for_centroids_kernel");

    // c. Finalize new centroids and write them into d_centroids (in-place update)
    finalize_centroids_kernel<<<init_finalize_grid_dim, init_finalize_block_dim>>>(
        d_temp_centroid_sums, d_temp_cluster_counts, d_centroids, K, D);
    CHECK_KERNEL_LAUNCH("finalize_centroids_kernel");
}