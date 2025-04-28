import numpy as np
import time

def counting_sort_byte_step(arr: np.ndarray, bit_shift: int):
    n = arr.size
    if n == 0:
        return

    output = np.zeros(n, dtype=np.uint32)
    count = np.zeros(256, dtype=np.intp)
    byte_mask = 0xFF

    for i in range(n):
        byte_value = (arr[i] >> bit_shift) & byte_mask
        count[byte_value] += 1

    for i in range(1, 256):
        count[i] += count[i-1]

    for i in range(n - 1, -1, -1):
        value = arr[i]
        byte_value = (value >> bit_shift) & byte_mask
        target_pos = count[byte_value] - 1
        output[target_pos] = value
        count[byte_value] -= 1

    arr[:] = output

def radix_sort(arr: np.ndarray):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    n = arr.size
    if n <= 1:
        return arr

    if arr.dtype != np.uint32:
        try:
            min_val = np.min(arr) if n > 0 else 0
            if min_val < 0:
                raise ValueError("Input array contains negative numbers, cannot use uint32 radix sort.")
            arr = arr.astype(np.uint32, copy=True)
        except ValueError as e:
            raise e

    bits_in_type = arr.itemsize * 8
    bits_per_pass = 8
    num_passes = (bits_in_type + bits_per_pass - 1) // bits_per_pass

    for pass_num in range(num_passes):
        bit_shift = pass_num * bits_per_pass
        counting_sort_byte_step(arr, bit_shift)

    return arr

if __name__ == "__main__":
    array_size = 2_000_000

    print(f"Generating {array_size:,} uint32 random integers (0 to {2**32 - 1})...")
    upper_bound = 2**32
    test_arr_orig = np.random.randint(0, upper_bound, size=array_size, dtype=np.uint32)
    test_arr_to_sort = test_arr_orig.copy()

    print(f"Starting single-threaded byte-based radix sort (using 'radix_sort' function)...")
    start_time = time.time()

    radix_sort(test_arr_to_sort)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Single-threaded radix sort complete in {elapsed_time:.3f} seconds.")
    if elapsed_time > 0:
         mb_per_sec = (test_arr_to_sort.nbytes / (1024**2)) / elapsed_time
         print(f"Throughput: {mb_per_sec:.2f} MB/s")

    print("\nVerifying sort result against NumPy sort...")
    start_time_np = time.time()
    correct_sorted_arr = np.sort(test_arr_orig)
    end_time_np = time.time()
    print(f"NumPy sort took {end_time_np - start_time_np:.3f} seconds.")

    if np.array_equal(test_arr_to_sort, correct_sorted_arr):
        print("Verification PASSED.")
    else:
        print("Verification FAILED.")
        diff_indices = np.where(test_arr_to_sort != correct_sorted_arr)[0]
        print(f"Mismatch count: {len(diff_indices)}")
        if len(diff_indices) > 0:
            first_diff = diff_indices[0]
            print(f"First mismatch at index {first_diff}:")
            print(f"  Original: {test_arr_orig[first_diff]}")
            print(f"  Got:      {test_arr_to_sort[first_diff]}")
            print(f"  Expected: {correct_sorted_arr[first_diff]}")
            start_idx = max(0, first_diff - 2)
            end_idx = min(array_size, first_diff + 3)
            print("Context around mismatch:")
            print(f"  Indices:  {np.arange(start_idx, end_idx)}")
            print(f"  Original: {test_arr_orig[start_idx:end_idx]}")
            print(f"  Got:      {test_arr_to_sort[start_idx:end_idx]}")
            print(f"  Expected: {correct_sorted_arr[start_idx:end_idx]}")
