#include <cuda_runtime.h>
#include <stdio.h>      // For printf, fprintf
#include <stdint.h>     // For uint32_t
#include <stdlib.h>     // For malloc/free
#include <utility>      // For std::swap
#include <stdexcept>    // For std::exception

// --- Thrust Includes ---
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h> // For thrust::device

#define DLL_EXPORT __declspec(dllexport)

// --- Configuration ---
#define SORT_BLOCK_SZ 128   // Threads per block (can be tuned)
#define RADIX_BITS 8        // Process 8 bits (1 byte) per pass
#define NUM_BUCKETS (1 << RADIX_BITS) // 2^8 = 256 buckets

// --- Helper Macros for Error Checking ---
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


// --- Kernels ---

/*
 * Kernel 1: Counts elements per byte bucket (256 buckets) within blocks,
 * calculates local offsets using shared memory atomics.
 * Processes 8 bits per pass (Radix-256).
 */
__global__ void radix_sort_pass_byte(uint32_t* d_temp_values,        // Output: Copy of input values for this pass
                                     uint32_t* d_local_offsets,      // Output: Local offset within the bucket group
                                     uint32_t* d_block_counts,       // Output: Counts per bucket/block, interleaved [B0D0..B0D255, B1D0..B1D255, ...]
                                     const uint32_t* d_current_in,   // Input: Data for this pass
                                     unsigned int shift,             // Current bit shift (0, 8, 16, 24)
                                     unsigned int n)                  // Total number of elements
{
    // Shared memory for block-wide histogram and local offset calculation
    __shared__ uint32_t s_counts[NUM_BUCKETS];      // Histogram for this block
    __shared__ uint32_t s_offsets[NUM_BUCKETS];     // Starting offset + atomic counter for local ranks

    unsigned int thid = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int block_dim = blockDim.x; // Should be SORT_BLOCK_SZ
    unsigned int grid_dim = gridDim.x;
    unsigned int gidx = block_id * block_dim + thid;

    // Initialize shared memory (only needs to be done once per block)
    // Use first NUM_BUCKETS threads for efficiency if block_dim >= NUM_BUCKETS
    if (thid < NUM_BUCKETS) {
        s_counts[thid] = 0;
        s_offsets[thid] = 0; // Will be initialized later by thread 0 after scan
    }
    // If block_dim < NUM_BUCKETS, remaining elements need initialization
    if (block_dim < NUM_BUCKETS) {
         for (int i = block_dim + thid; i < NUM_BUCKETS; i += block_dim) {
              s_counts[i] = 0;
              s_offsets[i] = 0;
         }
    }
    __syncthreads(); // Ensure shared memory is zeroed

    // 1. Load data and determine bucket index (byte value)
    uint32_t value = 0;
    uint32_t bucket_idx = 0; // 0 to 255
    bool is_valid = (gidx < n);

    if (is_valid) {
        value = d_current_in[gidx];
        // Extract byte using mask 0xFF (255)
        bucket_idx = (value >> shift) & (NUM_BUCKETS - 1);
        d_temp_values[gidx] = value; // Store original value for shuffle

        // --- 2. Build block histogram using atomics ---
        atomicAdd(&s_counts[bucket_idx], 1);
    }
    __syncthreads(); // Ensure all threads have contributed to the histogram

    // --- 3. Calculate local offsets within the block ---
    // Thread 0 calculates the exclusive scan of the block histogram
    // to get the starting offset for each bucket *within this block*.
    // It also copies the raw counts to global memory.
    if (thid == 0) {
        uint32_t run_sum = 0;
        for (int i = 0; i < NUM_BUCKETS; ++i) {
            uint32_t count = s_counts[i]; // Read count for this bucket
            s_offsets[i] = run_sum;       // Write starting offset (exclusive scan result)
            run_sum += count;             // Accumulate for next offset

            // Write the raw count to global memory for the global scan
            unsigned int count_idx = block_id * NUM_BUCKETS + i;
            if (count_idx < grid_dim * NUM_BUCKETS) { // Bounds check
                d_block_counts[count_idx] = count;
            }
        }
    }
    __syncthreads(); // Ensure scan is done and s_offsets are initialized

    // --- 4. Calculate and store local offset using atomics ---
    // Each thread atomically increments the counter for its bucket
    // and uses the value *before* the increment as its local rank.
    if (is_valid) {
        // atomicAdd returns the OLD value at the address before adding 1
        uint32_t local_rank = atomicAdd(&s_offsets[bucket_idx], 1);
        // The final local offset is the rank within the bucket (local_rank)
        d_local_offsets[gidx] = local_rank;
    }
    // No sync needed here as atomics handle synchronization per bucket
}


/*
 * Kernel 2: Shuffles elements based on local offsets and scanned global counts (8-bit version).
 */
__global__ void radix_shuffle_byte(uint32_t* d_current_out,           // Output: Destination array for this pass
                                   const uint32_t* d_temp_values,     // Input: Original values for this pass
                                   const uint32_t* d_local_offsets,   // Input: Local offset within the bucket group
                                   const uint32_t* d_scanned_counts,  // Input: Global scanned counts (offsets) [B0D0..B0D255, B1D0..B1D255, ...]
                                   unsigned int shift,                // Current bit shift
                                   unsigned int n)                    // Total number of elements
{
    unsigned int thid = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int block_dim = blockDim.x;
    unsigned int grid_dim = gridDim.x;
    unsigned int gidx = block_id * block_dim + thid;

    if (gidx < n) {
        uint32_t value = d_temp_values[gidx];
        uint32_t local_offset = d_local_offsets[gidx];
        // determine the 8-bit bucket index again
        uint32_t bucket_idx = (value >> shift) & (NUM_BUCKETS - 1); // Mask is 0xFF

        // read the global starting offset for this block and this bucket
        // index into scanned counts: block_id * NUM_BUCKETS + bucket_idx
        unsigned int count_idx = block_id * NUM_BUCKETS + bucket_idx;
        unsigned int global_offset_start = 0;
        unsigned int counts_len = grid_dim * NUM_BUCKETS;
         if (count_idx < counts_len) { // Bounds check read from scanned counts
             global_offset_start = d_scanned_counts[count_idx];
         } else {
             // Error condition
             // printf("ERROR: Shuffle read OOB from scanned_counts! count_idx=%u, max_idx=%u\n", count_idx, counts_len);
             return;
         }

        // calculate final destination position: Global start offset + Local offset within group
        unsigned int final_pos = global_offset_start + local_offset;

        // Write value to final position (with bounds check)
        if (final_pos < n) {
            d_current_out[final_pos] = value;
        } else {
            // error condition
            // printf("ERROR: Shuffle write OOB! gidx=%u, val=%u, bucket=%u, loc_off=%u, glob_off_idx=%u, glob_off=%u, final_pos=%u, n=%u\n",
            //        gidx, value, bucket_idx, local_offset, count_idx, global_offset_start, final_pos, n);
        }
    }
}


// --- Host Function (DLL Export) ---
/*
 * Manages the 8-bit (byte) radix sort process using the kernels above and Thrust scan.
 */
extern "C" DLL_EXPORT void radix_sort_byte(uint32_t* d_input, uint32_t* d_output, int n_int) {
    if (n_int <= 0) {
        fprintf(stderr, "Warning: radix_sort_byte called with n <= 0.\n");
        return;
    }
    unsigned int n = (unsigned int)n_int;

    // --- Device Memory Allocations ---
    uint32_t* d_temp_values = nullptr;
    uint32_t* d_local_offsets = nullptr;
    uint32_t* d_block_counts = nullptr;   // counts per bucket per block (interleaved)
    uint32_t* d_scanned_counts = nullptr; // global offsets after scanning block counts

    unsigned int block_sz = SORT_BLOCK_SZ;
    unsigned int grid_sz = (n + block_sz - 1) / block_sz;
    // size of the block counts array (NUM_BUCKETS entries per block)
    unsigned int counts_len = grid_sz * NUM_BUCKETS; // 256 * grid_sz

    // Allocate temporary buffers
    CHECK_CUDA_API(cudaMalloc(&d_temp_values, n * sizeof(uint32_t)), "Malloc d_temp_values");
    CHECK_CUDA_API(cudaMalloc(&d_local_offsets, n * sizeof(uint32_t)), "Malloc d_local_offsets");
    CHECK_CUDA_API(cudaMalloc(&d_block_counts, counts_len * sizeof(uint32_t)), "Malloc d_block_counts");
    CHECK_CUDA_API(cudaMalloc(&d_scanned_counts, counts_len * sizeof(uint32_t)), "Malloc d_scanned_counts");

    // --- Radix Sort Passes ---
    uint32_t* d_current_in = d_input;
    uint32_t* d_current_out = d_output;
    int num_bits_total = sizeof(uint32_t) * 8; // 32 bits
    // process 8 bits at a time
    int num_passes = (num_bits_total + RADIX_BITS - 1) / RADIX_BITS; // 32/8 = 4 passes

    // loop through each byte, from LSB to MSB
    for (int pass = 0; pass < num_passes; ++pass) {
        unsigned int shift = pass * RADIX_BITS; // 0, 8, 16, 24
        d_current_in = (pass % 2 == 0) ? d_input : d_output;
        d_current_out = (pass % 2 == 0) ? d_output : d_input;

        // --- Step 1: Count elements per bucket and calculate local offsets ---
        CHECK_CUDA_API(cudaMemset(d_block_counts, 0, counts_len * sizeof(uint32_t)), "Memset d_block_counts");
        // Launch the byte-based pass kernel
        radix_sort_pass_byte<<<grid_sz, block_sz>>>(
            d_temp_values,
            d_local_offsets,
            d_block_counts,  // Output: Counts per bucket/block
            d_current_in,
            shift,
            n);
        CHECK_KERNEL_LAUNCH("radix_sort_pass_byte");

        // --- Step 2: Scan global block counts using Thrust ---
        try {
            thrust::device_ptr<uint32_t> d_block_counts_ptr(d_block_counts);
            thrust::device_ptr<uint32_t> d_scanned_counts_ptr(d_scanned_counts);

            // Perform exclusive scan on the interleaved block counts array
            thrust::exclusive_scan(thrust::device,
                                   d_block_counts_ptr,
                                   d_block_counts_ptr + counts_len,
                                   d_scanned_counts_ptr);

             CHECK_CUDA_API(cudaDeviceSynchronize(), "Sync after Thrust scan");

        } catch (const std::exception& e) {
             fprintf(stderr, "Thrust Exception during scan: %s\n", e.what());
             // Cleanup
             cudaFree(d_scanned_counts); cudaFree(d_block_counts);
             cudaFree(d_local_offsets); cudaFree(d_temp_values);
             return;
        }
        // --- End Thrust Scan ---

        // --- Step 3: Shuffle elements ---
        // Launch the byte-based shuffle kernel
        radix_shuffle_byte<<<grid_sz, block_sz>>>(
            d_current_out,
            d_temp_values,
            d_local_offsets,
            d_scanned_counts,   // Pass the single scanned counts array
            shift,
            n);
        CHECK_KERNEL_LAUNCH("radix_shuffle_byte");

    } // End passes loop

    // --- Final Result ---
    // After 4 passes (an even number), the fully sorted result
    // is in the buffer originally pointed to by d_input.
    // The Python code expects the result in d_input.

    // --- Cleanup ---
    CHECK_CUDA_API(cudaFree(d_scanned_counts), "Free d_scanned_counts");
    CHECK_CUDA_API(cudaFree(d_block_counts), "Free d_block_counts");
    CHECK_CUDA_API(cudaFree(d_local_offsets), "Free d_local_offsets");
    CHECK_CUDA_API(cudaFree(d_temp_values), "Free d_temp_values");
}
