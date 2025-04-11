#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define DLL_EXPORT __declspec(dllexport)
#define BITS_PER_PASS 4
#define BUCKETS (1 << BITS_PER_PASS)
#define THREADS 256

extern "C" DLL_EXPORT void radix_sort(uint32_t* d_input, uint32_t* d_output, int n);

__global__ void histogramKernel(uint32_t* input, int* histo, int shift, int n) {
    __shared__ int localHist[BUCKETS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x < BUCKETS) localHist[threadIdx.x] = 0;
    __syncthreads();
    if (tid < n) {
        int bucket = (input[tid] >> shift) & (BUCKETS - 1);
        atomicAdd(&localHist[bucket], 1);
    }
    __syncthreads();
    if (threadIdx.x < BUCKETS)
        atomicAdd(&histo[threadIdx.x], localHist[threadIdx.x]);
}

__global__ void scanKernel(int* histo, int* scan) {
    __shared__ int temp[BUCKETS];
    int tid = threadIdx.x;
    if (tid < BUCKETS) temp[tid] = histo[tid];
    __syncthreads();
    for (int offset = 1; offset < BUCKETS; offset <<= 1) {
        int val = 0;
        if (tid >= offset) val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
    if (tid < BUCKETS)
        scan[tid] = tid == 0 ? 0 : temp[tid - 1];
}

__global__ void scatterKernel(uint32_t* input, uint32_t* output, int* scan, int shift, int n) {
    __shared__ int localOffsets[BUCKETS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x < BUCKETS) localOffsets[threadIdx.x] = 0;
    __syncthreads();

    int bucket = 0;
    if (tid < n) {
        bucket = (input[tid] >> shift) & (BUCKETS - 1);
        int pos = atomicAdd(&localOffsets[bucket], 1);
        output[scan[bucket] + pos] = input[tid];
    }
}

extern "C" void radix_sort(uint32_t* d_input, uint32_t* d_output, int n) {
    int* d_hist;
    int* d_scan;
    cudaMalloc(&d_hist, BUCKETS * sizeof(int));
    cudaMalloc(&d_scan, BUCKETS * sizeof(int));

    for (int shift = 0; shift < 32; shift += BITS_PER_PASS) {
        cudaMemset(d_hist, 0, BUCKETS * sizeof(int));
        histogramKernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(d_input, d_hist, shift, n);
        scanKernel<<<1, BUCKETS>>>(d_hist, d_scan);
        scatterKernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(d_input, d_output, d_scan, shift, n);
        std::swap(d_input, d_output);
    }

    cudaFree(d_hist);
    cudaFree(d_scan);
}
