import numpy as np
import cupy as cp
from ctypes import cdll, c_uint32, POINTER, c_int

lib = cdll.LoadLibrary("./libradixsort.dll")

lib.radix_sort.argtypes = [POINTER(c_uint32), POINTER(c_uint32), c_int]
lib.radix_sort.restype = None

def radix_sort_gpu(arr):
    """Calls the GPU-accelerated radix sort from the DLL."""
    arr_cpu = arr.get()
    arr_cpu = arr_cpu.astype(np.uint32)
    
    output_cpu = np.zeros_like(arr_cpu, dtype=np.uint32)
    
    input_ptr = arr_cpu.ctypes.data_as(POINTER(c_uint32))
    output_ptr = output_cpu.ctypes.data_as(POINTER(c_uint32))

    lib.radix_sort(input_ptr, output_ptr, arr_cpu.size)

    return cp.array(output_cpu)  # Convert back to CuPy array

# Test the radix_sort_gpu function
if __name__ == "__main__":
    # Generate a random array with integers
    arr = cp.random.randint(1, 100000, 10000, dtype=cp.uint32)
    print("Original array:", arr)
    
    # Sort using the GPU-accelerated radix sort from the DLL
    sorted_arr = radix_sort_gpu(arr)
    
    print("Sorted array:", sorted_arr)
    print("GPU-accelerated radix sort complete.")
