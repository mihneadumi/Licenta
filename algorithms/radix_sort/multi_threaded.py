import numpy as np
import time

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: Numba not found. Parallel execution will be limited.")
    def njit(parallel=False, cache=False):
        def decorator(func):
            return func
        return decorator
    prange = range
    NUMBA_AVAILABLE = False


# --- Numba accelerated counting sort pass ---
@njit(parallel=True, cache=True)
def counting_sort_pass_numba(arr: np.ndarray, bit_shift: int, n: int):
    if n == 0:
        return np.zeros(0, dtype=np.uint32)

    output = np.zeros(n, dtype=np.uint32)
    byte_mask = np.uint32(0xFF)
    num_buckets = 256

    # --- Parallel Counting ---
    num_threads = numba.get_num_threads() if NUMBA_AVAILABLE else 1
    local_counts = np.zeros((num_threads, num_buckets), dtype=np.intp)

    for i in prange(n):
        thread_id = numba.get_thread_id() if NUMBA_AVAILABLE else 0
        byte_value = (arr[i] >> bit_shift) & byte_mask
        local_counts[thread_id, byte_value] += 1

    # --- Serial Aggregation & Prefix Sum ---
    global_count = np.zeros(num_buckets, dtype=np.intp)
    for t in range(num_threads):
        for b in range(num_buckets):
            global_count[b] += local_counts[t, b]

    exclusive_scan = np.zeros(num_buckets, dtype=np.intp)
    for i in range(1, num_buckets):
        exclusive_scan[i] = exclusive_scan[i-1] + global_count[i-1]

    # --- Parallel Placement (Attempt) ---
    current_pos = exclusive_scan.copy()
    for i in range(n):
        value = arr[i]
        byte_value = (value >> bit_shift) & byte_mask
        target_pos = current_pos[byte_value]
        if target_pos < n:
             output[target_pos] = value
        current_pos[byte_value] += 1

    return output

# --- Main Radix Sort Function using Numba Pass ---
def radix_sort_numba(arr: np.ndarray):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    n = arr.size
    if n <= 1:
        return arr

    original_dtype = arr.dtype
    needs_copy = False
    if arr.dtype != np.uint32:
        try:
            min_val = np.min(arr) if n > 0 else 0
            if min_val < 0:
                raise ValueError("Input array contains negative numbers, cannot use uint32 radix sort.")
            arr = arr.astype(np.uint32, copy=True)
            needs_copy = True
        except ValueError as e:
            raise e

    bits_in_type = arr.itemsize * 8
    bits_per_pass = 8
    num_passes = (bits_in_type + bits_per_pass - 1) // bits_per_pass

    current_arr = arr

    for pass_num in range(num_passes):
        bit_shift = pass_num * bits_per_pass
        current_arr = counting_sort_pass_numba(current_arr, bit_shift, n)
        if current_arr is None or current_arr.size != n:
             raise RuntimeError(f"Numba sort pass {pass_num} failed.")

    if original_dtype != np.uint32:
         current_arr = current_arr.astype(original_dtype)

    return current_arr

# --- Example Usage ---
if __name__ == "__main__":
    array_size = 2_000_000

    print(f"Generating {array_size:,} uint32 random integers...")
    upper_bound = 2**32
    test_arr_orig = np.random.randint(0, upper_bound, size=array_size, dtype=np.uint32)
    test_arr_to_sort = test_arr_orig.copy()

    print(f"Starting Numba-accelerated multi-threaded radix sort...")
    print("Performing initial run (may include Numba compilation time)...")
    start_time = time.time()
    sorted_arr_numba_run1 = radix_sort_numba(test_arr_to_sort.copy())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Initial run completed in {elapsed_time:.3f} seconds.")

    print("\nPerforming second run (should use cached compiled code)...")
    start_time = time.time()
    sorted_arr_numba = radix_sort_numba(test_arr_to_sort)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Second run completed in {elapsed_time:.3f} seconds.")

    if elapsed_time > 0:
         elements_per_second = array_size / elapsed_time
         melements_per_second = elements_per_second / 1e6
         print(f"Throughput: {melements_per_second:.2f} MElements/s")

    print("\nVerifying sort result against NumPy sort...")
    start_time_np = time.time()
    correct_sorted_arr = np.sort(test_arr_orig)
    end_time_np = time.time()
    print(f"NumPy sort took {end_time_np - start_time_np:.3f} seconds.")

    if np.array_equal(sorted_arr_numba, correct_sorted_arr):
        print("Verification PASSED.")
    else:
        print("Verification FAILED.")