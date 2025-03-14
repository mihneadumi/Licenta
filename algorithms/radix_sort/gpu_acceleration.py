import cupy as cp

def counting_sort_gpu(arr, exp):
    """GPU-based counting sort for a specific digit (exp)."""
    n = arr.size
    output = cp.zeros_like(arr)
    count = cp.zeros(10, dtype=cp.int32)

    indices = (arr // exp) % 10
    count = cp.bincount(indices, minlength=10)

    count = cp.cumsum(count)

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    return output

def radix_sort_gpu(arr):
    """Performs GPU-accelerated Radix Sort using CuPy."""
    arr_gpu = cp.array(arr)  # Move array to GPU
    max_num = cp.max(arr_gpu)
    exp = 1

    while max_num // exp > 0:
        arr_gpu = counting_sort_gpu(arr_gpu, exp)
        exp *= 10

    return cp.asnumpy(arr_gpu)  # Move back to CPU

# test
if __name__ == "__main__":
    arr = cp.random.randint(1, 100000, 10000)
    print(arr)
    sorted_arr = radix_sort_gpu(arr)
    print(sorted_arr)
    print("GPU-accelerated radix sort complete.")
