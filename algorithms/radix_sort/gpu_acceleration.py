import time
import numpy as np
import cupy as cp
from ctypes import cdll, c_void_p, c_int, cast, POINTER, c_uint32
import os
import traceback

# --- Configuration ---
# Update DLL name and function name for the byte-based sort
dll_filename = "libsimpleradixsort.dll"
cuda_function_name = "radix_sort_byte"
# --- End Configuration ---

script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, dll_filename)

lib = None

try:
    if not os.path.exists(dll_path):
        dll_path_cwd = os.path.join(os.getcwd(), dll_filename)
        if os.path.exists(dll_path_cwd):
             dll_path = dll_path_cwd
        else:
             raise FileNotFoundError(f"DLL not found at {dll_path} or {dll_path_cwd}")
    lib = cdll.LoadLibrary(dll_path)
    print(f"Successfully loaded DLL: {dll_path}") # Keep load message
except (OSError, FileNotFoundError) as e:
    print(f"Error loading DLL '{dll_filename}': {e}")
    def radix_sort_gpu(arr_input: cp.ndarray) -> cp.ndarray:
         raise ImportError(f"radix_sort_gpu unavailable: Error loading {dll_filename}.")
except NameError:
     print("Error: ctypes or cdll seems unavailable.")
     raise

if lib:
    try:
        sort_func = getattr(lib, cuda_function_name)
        sort_func.argtypes = [POINTER(c_uint32), POINTER(c_uint32), c_int]
        sort_func.restype = None
        print(f"Successfully found function '{cuda_function_name}' in DLL.") # Keep function found message

        def radix_sort_gpu(arr_input: cp.ndarray) -> cp.ndarray:
            """
            Wrapper for the CUDA byte-based radix sort DLL function.
            Sorts the input CuPy array (expects result in input buffer).
            """
            if not isinstance(arr_input, cp.ndarray):
                raise TypeError("Input must be a CuPy ndarray.")
            if arr_input.dtype != np.uint32:
                # Ensure copy=True if type changes
                arr_input = arr_input.astype(np.uint32, copy=True)

            n = arr_input.size
            if n == 0:
                return arr_input

            # Create empty output buffer (required by C++ signature)
            arr_output_gpu = cp.empty_like(arr_input)

            try:
                input_ptr_gpu = cast(arr_input.data.ptr, POINTER(c_uint32))
                output_ptr_gpu = cast(arr_output_gpu.data.ptr, POINTER(c_uint32))
            except AttributeError:
                 print("Error: Could not get GPU data pointers using .data.ptr")
                 raise

            try:
                # Call the C++ byte sort function
                sort_func(input_ptr_gpu, output_ptr_gpu, n)
            except Exception as e:
                 print(f"Error during call to lib.{cuda_function_name}: {e}")
                 traceback.print_exc()
                 raise

            cp.cuda.Stream.null.synchronize()
            # Result is in the input buffer after even number of passes (4)
            return arr_input

    except AttributeError:
        print(f"Error: Function '{cuda_function_name}' not found in DLL.")
        def radix_sort_gpu(arr_input: cp.ndarray) -> cp.ndarray:
             raise ImportError(f"radix_sort_gpu unavailable: Function '{cuda_function_name}' not found in DLL.")

if 'radix_sort_gpu' not in locals():
     def radix_sort_gpu(arr_input: cp.ndarray) -> cp.ndarray:
         raise ImportError("radix_sort_gpu unavailable due to DLL/function loading error.")

if __name__ == "__main__":
    gpu_id = 0
    try:
        cp.cuda.Device(gpu_id).use()
        device_props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        gpu_name = device_props['name'].decode('utf-8')
        print(f"Using GPU: {gpu_name}") # Keep GPU info print
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CUDA Error initializing CuPy: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred initializing CuPy: {e}")
        traceback.print_exc()
        exit(1)

    # --- Test Parameters ---
    array_size = 50_000_000 # Use underscore for readability (e.g., 50 Million)
    min_val = 0
    max_val = 2**32 # Use full uint32 range for better test
    print(f"\nGenerating array of size {array_size:,}...")
    # --- End Test Parameters ---


    # Generate test data (once for all sorts)
    test_arr_np_orig = np.random.randint(min_val, max_val, size=array_size, dtype=np.uint32)
    print("Array generation complete.")

    try:
        # --- Time GPU Sort (DLL) ---
        print("\nStarting GPU Radix Sort (DLL)...")
        input_copy_gpu = cp.asarray(test_arr_np_orig) # Copy to GPU
        cp.cuda.Stream.null.synchronize() # Ensure copy is done
        start_time_gpu = time.time()
        sorted_arr_gpu = radix_sort_gpu(input_copy_gpu) # Sorts the CuPy array
        cp.cuda.Stream.null.synchronize() # Ensure sort is done
        end_time_gpu = time.time()
        gpu_time = end_time_gpu - start_time_gpu
        print(f"GPU sort time: {gpu_time:.3f} seconds")
        # Optional: Copy result back if needed for other comparisons
        # sorted_gpu_result_np = sorted_arr_gpu.get()
        # del sorted_arr_gpu # Free GPU memory if large
        # del input_copy_gpu
        # cp.get_default_memory_pool().free_all_blocks()


        # --- Time CuPy Sort ---
        print("\nStarting CuPy built-in sort...")
        test_arr_gpu = cp.asarray(test_arr_np_orig) # Fresh copy to GPU
        cp.cuda.Stream.null.synchronize()
        start_time_cupy = time.time()
        cupy_sorted = cp.sort(test_arr_gpu) # Use CuPy's sort
        cp.cuda.Stream.null.synchronize()
        end_time_cupy = time.time()
        cupy_time = end_time_cupy - start_time_cupy
        print(f"CuPy sort time: {cupy_time:.3f} seconds")
        # Optional: Copy result back
        # cupy_sorted_np = cupy_sorted.get()
        # del test_arr_gpu # Free GPU memory
        # del cupy_sorted
        # cp.get_default_memory_pool().free_all_blocks()


        # --- Time CPU Sort (NumPy) ---
        # Using np.sort is generally faster and more standard than list.sort() for NumPy arrays
        print("\nStarting CPU NumPy sort...")
        arr_to_sort_np = test_arr_np_orig.copy() # Work on a copy
        start_time_cpu_np = time.time()
        # np.sort returns a sorted copy
        sorted_arr_cpu_np = np.sort(arr_to_sort_np)
        end_time_cpu_np = time.time()
        cpu_np_time = end_time_cpu_np - start_time_cpu_np
        print(f"CPU NumPy sort time: {cpu_np_time:.3f} seconds")


        # --- Verification (Compare DLL result against NumPy result) ---
        # Note: Verification requires having both results available.
        # If memory is tight, you might need to re-run one of the sorts
        # or verify smaller arrays. For now, assume sorted_arr_gpu and
        # sorted_arr_cpu_np hold the results (potentially as NumPy arrays).
        # Ensure sorted_arr_gpu is copied back to host for comparison:
        print("\nVerifying DLL sort against NumPy sort...")
        sorted_gpu_result_np = sorted_arr_gpu.get() # Copy GPU result back

        if np.array_equal(sorted_gpu_result_np, sorted_arr_cpu_np):
            print("Verification PASSED: DLL sort matches NumPy sort.")
        else:
            print("Verification FAILED: DLL sort does NOT match NumPy sort.")
            # Add difference printing logic if needed (as before)

    except ImportError as e:
         print(f"\nCould not run test due to import error: {e}")
    except cp.cuda.memory.OutOfMemoryError:
        print("\nCUDA Out of Memory Error: Array size might be too large for GPU memory.")
        traceback.print_exc()
    except Exception as e:
        print(f"\nAn error occurred during the test execution: {e}")
        traceback.print_exc()

    print(f"\n--- Test complete ---")
