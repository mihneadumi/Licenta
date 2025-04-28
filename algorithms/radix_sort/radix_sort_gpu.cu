#include <cuda_runtime.h>
#include <stdio.h>      // For printf (error reporting), fprintf
#include <stdint.h>     // For uint32_t
#include <stdlib.h>     // For malloc/free in debug prints (if used)
#include <utility>      // Required for std::swap
#include <cmath>        // For log2f
#include <stdexcept>    // For std::exception in Thrust catch block

// --- Thrust Includes ---
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h> // For thrust::device

#define DLL_EXPORT __declspec(dllexport)

// --- Configuration ---
#define MAX_BLOCK_SZ 128
#define RADIX_BITS 2
#define NUM_BUCKETS (1 << RADIX_BITS) // 4
#define SCAN_BLOCK_SZ (MAX_BLOCK_SZ / 2) // 64 // Kept for reference, but scan kernels not used if Thrust works
#define SCAN_MAX_ELEMS_PER_BLOCK (2 * SCAN_BLOCK_SZ) // 128
#define NUM_BANKS 32 // Kept for reference
#define LOG_NUM_BANKS 5 // Kept for reference

// --- Padding Macros ---
// Kept for reference, not directly used by Thrust scan typically
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define PADDED_SIZE(n) ((n) + CONFLICT_FREE_OFFSET(n - 1))

// --- Helper Macros for Error Checking ---
// Macro to check CUDA API calls for errors
#define CHECK_CUDA_API(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA API Error at %s (%s:%d): %s\n", msg, __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* Consider returning an error code instead of exit */ \
            return; \
        } \
    } while (0)

// Macro to check for kernel launch errors and synchronize
#define CHECK_KERNEL_LAUNCH(msg) \
    do { \
        /* Check for errors from asynchronous launch */ \
        cudaError_t err = cudaPeekAtLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Launch Error before %s (%s:%d): %s\n", msg, __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
        /* Synchronize device to ensure kernel completion and check for runtime errors */ \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error after %s kernel sync (%s:%d): %s\n", msg, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)


// --- Kernels ---

/*
 * Kernel: Performs local radix sort using separate shared memory arrays.
 * (No changes from previous version - includes padding fix)
 */
__global__ void gpu_radix_sort_local(uint32_t* d_out_locally_sorted,
                                     uint32_t* d_prefix_sums,
                                     uint32_t* d_block_sums,
                                     unsigned int input_shift_width,
                                     const uint32_t* d_in,
                                     unsigned int d_in_len,
                                     unsigned int max_elems_per_block)
{
    // Separate shared memory arrays
    __shared__ uint32_t s_data[MAX_BLOCK_SZ];
    __shared__ uint32_t s_mask_out[MAX_BLOCK_SZ];
    __shared__ uint32_t s_merged_scan_mask_out[MAX_BLOCK_SZ];
    __shared__ uint32_t s_mask_out_sums[NUM_BUCKETS];
    __shared__ uint32_t s_scan_mask_out_sums[NUM_BUCKETS];

    unsigned int thid = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int grid_dim = gridDim.x;
    unsigned int block_dim = blockDim.x;

    unsigned int cpy_idx = max_elems_per_block * block_id + thid;
    uint32_t t_data = 0;
    bool is_valid_element = (cpy_idx < d_in_len);

    // Copy input to shared memory, padding invalid elements
    if (is_valid_element) {
        t_data = d_in[cpy_idx];
        s_data[thid] = t_data;
    } else {
        s_data[thid] = 0; // Pad with 0
    }

    // Zero out intermediate shared arrays
    if (thid < block_dim) s_merged_scan_mask_out[thid] = 0;
    if (thid < NUM_BUCKETS) {
        s_mask_out_sums[thid] = 0;
        s_scan_mask_out_sums[thid] = 0;
    }
    __syncthreads(); // Ensure initialization is complete

    // Extract the 2-bit digit for the current pass
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & (NUM_BUCKETS - 1);

    // Iterate through each possible digit value (0, 1, 2, 3)
    for (unsigned int i = 0; i < NUM_BUCKETS; ++i) {
        // Zero out the mask array for this digit pass
        if (thid < block_dim) s_mask_out[thid] = 0;
        __syncthreads();

        // Build the bit mask: 1 if the element's digit matches 'i' AND element is valid
        bool val_equals_i = false;
        if (is_valid_element) {
            val_equals_i = (t_2bit_extract == i);
            s_mask_out[thid] = val_equals_i ? 1 : 0;
        } else {
            s_mask_out[thid] = 0; // Ensure padded elements do not contribute
        }
        __syncthreads(); // Ensure mask is built before scanning

        // Perform an inclusive scan (Hillis-Steele) on the mask in shared memory
        for (unsigned int offset = 1; offset < block_dim; offset *= 2) {
            unsigned int temp = 0;
            if (thid >= offset) temp = s_mask_out[thid - offset];
            __syncthreads();
            if (thid >= offset) s_mask_out[thid] += temp;
            __syncthreads();
        }

        // Get the total count for this digit in this block
        unsigned int total_sum_for_digit = 0;
        if (block_dim > 0) total_sum_for_digit = s_mask_out[block_dim - 1];

        // Thread 0 writes the total sum to shared memory and global block sums array
        if (thid == 0) {
            s_mask_out_sums[i] = total_sum_for_digit;
            d_block_sums[i * grid_dim + block_id] = total_sum_for_digit;
        }

        // Convert inclusive scan result to exclusive scan value
        unsigned int exclusive_scan_val = (thid == 0) ? 0 : s_mask_out[thid - 1];
        __syncthreads();

        // Store the exclusive scan value (local prefix sum) if this element belongs to digit 'i'
        if (val_equals_i && is_valid_element) {
             s_merged_scan_mask_out[thid] = exclusive_scan_val;
        }
        __syncthreads();
    } // End loop over digits

    // Scan the local digit counts stored in s_mask_out_sums (serial scan by thread 0)
    if (thid == 0) {
        unsigned int run_sum = 0;
        for (unsigned int k = 0; k < NUM_BUCKETS; ++k) {
            s_scan_mask_out_sums[k] = run_sum;
            run_sum += s_mask_out_sums[k];
        }
    }
    __syncthreads();

    // Calculate final local position and shuffle data within the block
    if (is_valid_element) {
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
        unsigned int bucket_start_offset = s_scan_mask_out_sums[t_2bit_extract];
        unsigned int new_pos = t_prefix_sum + bucket_start_offset;

        // Write the local prefix sum to global memory
        d_prefix_sums[cpy_idx] = t_prefix_sum;

        // Read original data before shuffle
        uint32_t original_data = s_data[thid];
        __syncthreads();

        // Shuffle data into the new local position
        if (new_pos < block_dim) {
            s_data[new_pos] = original_data;
        }
        __syncthreads();

        // Write locally sorted data back to global temp buffer
        d_out_locally_sorted[cpy_idx] = s_data[thid];

    } else {
         // Handle padding
         d_prefix_sums[cpy_idx] = 0;
         d_out_locally_sorted[cpy_idx] = 0;
    }
}


/*
 * Kernel: Global shuffle (Scatter).
 * (No changes from previous version - includes atomic counter)
 */
__global__ void gpu_glbl_shuffle(uint32_t* d_out,
                                 const uint32_t* d_in_locally_sorted,
                                 const uint32_t* d_scanned_block_sums,
                                 const uint32_t* d_prefix_sums,
                                 unsigned int input_shift_width,
                                 unsigned int d_in_len,
                                 unsigned int max_elems_per_block,
                                 unsigned int* d_oob_counter)
{
    unsigned int thid = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int grid_dim = gridDim.x;
    unsigned int cpy_idx = max_elems_per_block * block_id + thid;

    if (cpy_idx < d_in_len) {
        uint32_t t_data = d_in_locally_sorted[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & (NUM_BUCKETS - 1);
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];

        unsigned int block_sum_offset_idx = t_2bit_extract * grid_dim + block_id;
        unsigned int block_sum_offset = 0;
        unsigned int scan_buffer_len = NUM_BUCKETS * grid_dim;
        if (block_sum_offset_idx < scan_buffer_len) {
            block_sum_offset = d_scanned_block_sums[block_sum_offset_idx];
        } else {
             atomicAdd(d_oob_counter, 1); // Index itself is OOB
        }

        unsigned int data_glbl_pos = block_sum_offset + t_prefix_sum;

        if (data_glbl_pos >= d_in_len) {
             atomicAdd(d_oob_counter, 1); // Calculated position is OOB
        } else {
             d_out[data_glbl_pos] = t_data; // Write if in bounds
        }
    }
}

// --- Host Function ---

/*
 * Host function: Manages the overall radix sort process.
 * MODIFIED TO USE THRUST::EXCLUSIVE_SCAN for global scan.
 * Custom scan kernels (gpu_prescan, gpu_add_block_sums) and host function
 * (sum_scan_blelloch) are no longer called but kept for reference if needed later.
 */
extern "C" DLL_EXPORT void radix_sort(uint32_t* d_input, uint32_t* d_output, int n) {
    // Handle invalid input size
    if (n <= 0) {
        fprintf(stderr, "Warning: radix_sort called with n <= 0.\n");
        return;
    }
    unsigned int d_in_len = (unsigned int)n;

    // --- Kernel Launch Configuration ---
    unsigned int sort_block_sz = MAX_BLOCK_SZ;
    unsigned int max_elems_per_block_sort = sort_block_sz;
    unsigned int grid_sz = (d_in_len + max_elems_per_block_sort - 1) / max_elems_per_block_sort;
     if (grid_sz == 0 && d_in_len > 0) grid_sz = 1;

    // --- Temporary Device Memory Allocation ---
    uint32_t* d_temp_buffer = nullptr;
    uint32_t* d_prefix_sums = nullptr;
    uint32_t* d_block_sums = nullptr;
    uint32_t* d_scan_block_sums = nullptr;
    unsigned int* d_oob_counter = nullptr;
    unsigned int h_oob_counter = 0;

    CHECK_CUDA_API(cudaMalloc(&d_temp_buffer, d_in_len * sizeof(uint32_t)), "Malloc d_temp_buffer");
    CHECK_CUDA_API(cudaMalloc(&d_prefix_sums, d_in_len * sizeof(uint32_t)), "Malloc d_prefix_sums");
    unsigned int d_block_sums_len = NUM_BUCKETS * grid_sz;
    CHECK_CUDA_API(cudaMalloc(&d_block_sums, d_block_sums_len * sizeof(uint32_t)), "Malloc d_block_sums");
    CHECK_CUDA_API(cudaMalloc(&d_scan_block_sums, d_block_sums_len * sizeof(uint32_t)), "Malloc d_scan_block_sums");
    CHECK_CUDA_API(cudaMalloc(&d_oob_counter, sizeof(unsigned int)), "Malloc d_oob_counter");
    CHECK_CUDA_API(cudaMemset(d_oob_counter, 0, sizeof(unsigned int)), "Memset d_oob_counter");

    // Calculate shared memory size needed for gpu_radix_sort_local kernel
    unsigned int local_sort_shmem_elements = (3 * MAX_BLOCK_SZ) + (2 * NUM_BUCKETS);
    unsigned int local_sort_shmem_sz_bytes = local_sort_shmem_elements * sizeof(uint32_t);

    // --- Radix Sort Passes ---
    uint32_t* d_current_in = d_input;
    uint32_t* d_current_out = d_output;
    int num_bits_total = sizeof(uint32_t) * 8;
    int num_passes = (num_bits_total + RADIX_BITS - 1) / RADIX_BITS;

    for (unsigned int pass = 0; pass < num_passes; ++pass) {
         unsigned int shift_width = pass * RADIX_BITS;
         d_current_in = (pass % 2 == 0) ? d_input : d_output;
         d_current_out = (pass % 2 == 0) ? d_output : d_input;

        // --- Step 1: Local Radix Sort ---
        CHECK_CUDA_API(cudaMemset(d_block_sums, 0, d_block_sums_len * sizeof(uint32_t)), "Memset d_block_sums for pass");
        gpu_radix_sort_local<<<grid_sz, sort_block_sz, local_sort_shmem_sz_bytes>>>(
            d_temp_buffer, d_prefix_sums, d_block_sums, shift_width,
            d_current_in, d_in_len, max_elems_per_block_sort);
        CHECK_KERNEL_LAUNCH("gpu_radix_sort_local");

        // --- Step 2: Scan Global Block Sums (Using Thrust) ---
        try {
            // Create device pointers for Thrust
            thrust::device_ptr<uint32_t> d_block_sums_ptr(d_block_sums);
            thrust::device_ptr<uint32_t> d_scan_block_sums_ptr(d_scan_block_sums);

            // Perform exclusive scan using Thrust on the default stream
            thrust::exclusive_scan(thrust::device, // Use default CUDA stream
                                   d_block_sums_ptr,
                                   d_block_sums_ptr + d_block_sums_len, // End iterator
                                   d_scan_block_sums_ptr);              // Output iterator

             // Synchronize after Thrust call to ensure completion before shuffle uses the result
             CHECK_CUDA_API(cudaDeviceSynchronize(), "Sync after Thrust scan");

        } catch (const std::exception& e) {
            // Catch potential Thrust exceptions
            fprintf(stderr, "Thrust Exception during scan: %s\n", e.what());
            // Cleanup and return on error
             if (d_oob_counter) cudaFree(d_oob_counter);
             if (d_scan_block_sums) cudaFree(d_scan_block_sums);
             if (d_block_sums) cudaFree(d_block_sums);
             if (d_prefix_sums) cudaFree(d_prefix_sums);
             if (d_temp_buffer) cudaFree(d_temp_buffer);
             return;
        }
        // --- End Thrust Scan ---

        // --- Step 3: Global Shuffle (Scatter) ---
        gpu_glbl_shuffle<<<grid_sz, sort_block_sz>>>(
            d_current_out, d_temp_buffer, d_scan_block_sums, d_prefix_sums,
            shift_width, d_in_len, max_elems_per_block_sort, d_oob_counter);
        CHECK_KERNEL_LAUNCH("gpu_glbl_shuffle");

    } // End passes loop

    // --- Check Atomic Counter ---
    CHECK_CUDA_API(cudaDeviceSynchronize(), "Sync before reading OOB counter");
    CHECK_CUDA_API(cudaMemcpy(&h_oob_counter, d_oob_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost), "Memcpy OOB counter D2H");
    fprintf(stderr, "[INFO] Radix sort shuffle detected %u out-of-bounds write attempt(s).\n", h_oob_counter);

    // --- Final Result --- (Result is in d_input after 16 passes)

    // --- Free Temporary Device Memory ---
    if (d_oob_counter) CHECK_CUDA_API(cudaFree(d_oob_counter), "Free d_oob_counter");
    if (d_scan_block_sums) CHECK_CUDA_API(cudaFree(d_scan_block_sums), "Free d_scan_block_sums");
    if (d_block_sums) CHECK_CUDA_API(cudaFree(d_block_sums), "Free d_block_sums");
    if (d_prefix_sums) CHECK_CUDA_API(cudaFree(d_prefix_sums), "Free d_prefix_sums");
    if (d_temp_buffer) CHECK_CUDA_API(cudaFree(d_temp_buffer), "Free d_temp_buffer");
}

/*
 * NOTE: The custom scan functions (sum_scan_blelloch, gpu_prescan, gpu_add_block_sums)
 * are no longer called in this version but are left here for reference
 * in case Thrust cannot be used or further debugging of the custom scan is needed.
 */

/*
 * Kernel: Performs Blelloch scan (Exclusive).
 * (No changes from previous correct version - includes corrected downsweep)
 * --- NOT CALLED WHEN USING THRUST ---
 */
__global__ void gpu_prescan(unsigned int* d_out,
                            const unsigned int* d_in,
                            unsigned int* d_block_sums_aggregate,
                            const unsigned int len,
                            const unsigned int max_elems_per_block)
{
    // ... (implementation as before) ...
     extern __shared__ unsigned int s_mem[];
     int thid = threadIdx.x; int block_dim_scan = blockDim.x; int block_id = blockIdx.x; int grid_dim = gridDim.x;
     unsigned int shmem_size_elements = PADDED_SIZE(max_elems_per_block);
     int ai = thid; int bi = thid + block_dim_scan;
     unsigned int global_ai = max_elems_per_block * block_id + ai; unsigned int global_bi = max_elems_per_block * block_id + bi;
     int sai = ai + CONFLICT_FREE_OFFSET(ai); int sbi = bi + CONFLICT_FREE_OFFSET(bi);
     for (int i = thid; i < shmem_size_elements; i += block_dim_scan) { s_mem[i] = 0; } __syncthreads();
     if (global_ai < len) s_mem[sai] = d_in[global_ai]; if (global_bi < len) s_mem[sbi] = d_in[global_bi]; __syncthreads();
     int offset = 1;
     for (int d = max_elems_per_block >> 1; d > 0; d >>= 1) { __syncthreads(); if (thid < d) { int logical_a_idx = offset * (2 * thid + 1) - 1; int logical_b_idx = offset * (2 * thid + 2) - 1; int sidx_a = logical_a_idx + CONFLICT_FREE_OFFSET(logical_a_idx); int sidx_b = logical_b_idx + CONFLICT_FREE_OFFSET(logical_b_idx); if (sidx_a < shmem_size_elements && sidx_b < shmem_size_elements) { s_mem[sidx_b] += s_mem[sidx_a]; } } offset <<= 1; }
     if (thid == 0) { int last_logical_idx = max_elems_per_block - 1; int slast_elem_idx = last_logical_idx + CONFLICT_FREE_OFFSET(last_logical_idx); if (d_block_sums_aggregate != nullptr && block_id < grid_dim) { if (slast_elem_idx < shmem_size_elements) { d_block_sums_aggregate[block_id] = s_mem[slast_elem_idx]; } } if (slast_elem_idx < shmem_size_elements) { s_mem[slast_elem_idx] = 0; } }
     for (int offset_down = max_elems_per_block >> 1; offset_down > 0; offset_down >>= 1) { __syncthreads(); if (thid < (max_elems_per_block / (offset_down * 2))) { int logical_a_idx = offset_down * (2 * thid + 1) - 1; int logical_b_idx = offset_down * (2 * thid + 2) - 1; int sidx_a = logical_a_idx + CONFLICT_FREE_OFFSET(logical_a_idx); int sidx_b = logical_b_idx + CONFLICT_FREE_OFFSET(logical_b_idx); if (sidx_a < shmem_size_elements && sidx_b < shmem_size_elements) { unsigned int temp = s_mem[sidx_a]; s_mem[sidx_a] = s_mem[sidx_b]; s_mem[sidx_b] += temp; } } } __syncthreads();
     if (global_ai < len) d_out[global_ai] = s_mem[sai]; if (global_bi < len) d_out[global_bi] = s_mem[sbi];
}

/*
 * Kernel: Adds scanned block sums back to the elements within each block.
 * (No changes needed from previous version)
 * --- NOT CALLED WHEN USING THRUST ---
 */
__global__ void gpu_add_block_sums(uint32_t* d_out,
                                   const uint32_t* d_scanned_elements,
                                   const uint32_t* d_scanned_block_sums,
                                   const size_t numElems,
                                   const unsigned int max_elems_per_block)
{
    // ... (implementation as before) ...
    unsigned int block_id = blockIdx.x; unsigned int thid = threadIdx.x; unsigned int block_dim_scan = blockDim.x;
    unsigned int num_blocks = (unsigned int)((numElems + max_elems_per_block - 1) / max_elems_per_block); uint32_t block_sum_val = 0;
    if (block_id < num_blocks) { block_sum_val = d_scanned_block_sums[block_id]; }
    unsigned int idx1 = max_elems_per_block * block_id + thid; unsigned int idx2 = idx1 + block_dim_scan;
    if (idx1 < numElems) d_out[idx1] = d_scanned_elements[idx1] + block_sum_val; if (idx2 < numElems) d_out[idx2] = d_scanned_elements[idx2] + block_sum_val;
}

/*
 * Host function: Manages multi-level Blelloch scan.
 * (No changes needed from previous version)
 * --- NOT CALLED WHEN USING THRUST ---
 */
void sum_scan_blelloch(uint32_t* d_out,
                       const uint32_t* d_in,
                       const size_t numElems)
{
    // ... (implementation as before) ...
    if (numElems == 0) return;
    CHECK_CUDA_API(cudaMemset(d_out, 0, numElems * sizeof(uint32_t)), "Memset d_out in scan");
    unsigned int block_sz = SCAN_BLOCK_SZ; unsigned int max_elems_per_block = SCAN_MAX_ELEMS_PER_BLOCK;
    unsigned int grid_sz = (unsigned int)((numElems + max_elems_per_block - 1) / max_elems_per_block); if (grid_sz == 0 && numElems > 0) grid_sz = 1;
    unsigned int shmem_elements = PADDED_SIZE(max_elems_per_block); unsigned int shmem_sz_bytes = shmem_elements * sizeof(unsigned int);
    uint32_t* d_block_sums = nullptr; uint32_t* d_scanned_block_sums = nullptr;
    if (grid_sz > 1) { CHECK_CUDA_API(cudaMalloc(&d_block_sums, grid_sz * sizeof(uint32_t)), "Malloc d_block_sums"); CHECK_CUDA_API(cudaMemset(d_block_sums, 0, grid_sz * sizeof(uint32_t)), "Memset d_block_sums"); CHECK_CUDA_API(cudaMalloc(&d_scanned_block_sums, grid_sz * sizeof(uint32_t)), "Malloc d_scanned_block_sums"); }
    gpu_prescan<<<grid_sz, block_sz, shmem_sz_bytes>>>(d_out, d_in, d_block_sums, (unsigned int)numElems, max_elems_per_block);
    cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { fprintf(stderr, "CUDA Kernel Launch Error (gpu_prescan): %s\n", cudaGetErrorString(err)); return; }
    if (grid_sz > 1) { CHECK_CUDA_API(cudaDeviceSynchronize(), "Sync before recursive scan"); sum_scan_blelloch(d_scanned_block_sums, d_block_sums, grid_sz); err = cudaGetLastError(); if (err != cudaSuccess) { fprintf(stderr, "CUDA Error after recursive scan call: %s\n", cudaGetErrorString(err)); return; } gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_scanned_block_sums, numElems, max_elems_per_block); err = cudaGetLastError(); if (err != cudaSuccess) { fprintf(stderr, "CUDA Kernel Launch Error (gpu_add_block_sums): %s\n", cudaGetErrorString(err)); return; } CHECK_CUDA_API(cudaDeviceSynchronize(), "Sync after add_block_sums"); if (d_scanned_block_sums) CHECK_CUDA_API(cudaFree(d_scanned_block_sums), "Free d_scanned_block_sums"); if (d_block_sums) CHECK_CUDA_API(cudaFree(d_block_sums), "Free d_block_sums"); } else { CHECK_CUDA_API(cudaDeviceSynchronize(), "Sync after single block scan"); }
}
