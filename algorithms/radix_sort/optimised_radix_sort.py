import numpy as np


def counting_sort(arr, exp, byte_check, buckets):
    """Performs counting sort based on a specific byte (exp)."""
    n = len(arr)

    # Count occurrences of each byte
    count = np.zeros(256, dtype=int)
    for num in arr:
        byte_at_offset = (num & byte_check) >> exp
        count[byte_at_offset] += 1

    # Cumulative count for sorting
    for i in range(1, 256):
        count[i] += count[i - 1]

    # Place elements into correct buckets
    output = np.zeros(n, dtype=int)
    for num in reversed(arr):
        byte_at_offset = (num & byte_check) >> exp
        output[count[byte_at_offset] - 1] = num
        count[byte_at_offset] -= 1

    # Copy the sorted output back to the original array
    for i in range(n):
        arr[i] = output[i]


def radix_sort(arr):
    """Performs Radix Sort using bitwise operations and optimizations."""
    max_num = np.max(arr)
    max_bits = max(int(max_num).bit_length(), 1)  # Ensure at least 1 bit for empty array handling
    highest_byte = (max_bits + 7) // 8  # Calculate the number of bytes needed to represent max_num

    exp = 0
    byte_check = 0xFF  # Initial byte mask

    # Perform counting sort on each byte (LSB to MSB)
    for offset in range(highest_byte):
        counting_sort(arr, exp, byte_check, [])
        byte_check <<= 8  # Move the mask to the next byte
        exp += 8

    return arr


# Example usage
if __name__ == "__main__":
    arr = np.random.randint(1, 10000000, 1000000)
    print("Before Sorting:", arr[:10])  # Print first 10 elements before sorting
    radix_sort(arr)
    print("After Sorting:", arr[:10])  # Print first 10 elements after sorting
    print("Radix sort complete.")
