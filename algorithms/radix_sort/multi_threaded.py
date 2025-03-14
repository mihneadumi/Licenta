import numpy as np
from concurrent.futures import ThreadPoolExecutor

def counting_sort_parallel(arr, exp, num_threads=4):
    """Parallel counting sort based on a specific digit (exp)."""
    n = len(arr)
    output = np.zeros(n, dtype=int)
    count = np.zeros(10, dtype=int)

    def count_occurrences(start, end):
        local_count = np.zeros(10, dtype=int)
        for i in range(start, end):
            index = (arr[i] // exp) % 10
            local_count[index] += 1
        return local_count

    chunk_size = n // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda x: count_occurrences(x, min(x + chunk_size, n)), range(0, n, chunk_size))

    for local_count in results:
        count += local_count

    for i in range(1, 10):
        count[i] += count[i - 1]

    def place_elements(start, end):
        local_output = np.zeros(n, dtype=int)
        local_count = count.copy()
        for i in range(end - 1, start - 1, -1):
            index = (arr[i] // exp) % 10
            local_output[local_count[index] - 1] = arr[i]
            local_count[index] -= 1
        return local_output

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda x: place_elements(x, min(x + chunk_size, n)), range(0, n, chunk_size))

    for local_output in results:
        output += local_output

    for i in range(n):
        arr[i] = output[i]

def radix_sort_parallel(arr, num_threads=4):
    """Parallel Radix Sort."""
    max_num = np.max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort_parallel(arr, exp, num_threads)
        exp *= 10
    return arr

# Example usage
if __name__ == "__main__":
    arr = np.random.randint(1, 100000, 10000)
    sorted_arr = radix_sort_parallel(arr)
    print("Multi-threaded radix sort complete.")
