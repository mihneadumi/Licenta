import numpy as np

def counting_sort(arr, exp):
    """Performs counting sort based on a specific digit (exp)."""
    n = len(arr)
    output = np.zeros(n, dtype=int)
    count = np.zeros(10, dtype=int)

    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    """Performs Radix Sort."""
    max_num = np.max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr

# Example usage
if __name__ == "__main__":
    arr = np.random.randint(1, 100000, 10000)
    print(arr)
    radix_sort(arr)
    print(arr)
    print("Radix sort complete.")
