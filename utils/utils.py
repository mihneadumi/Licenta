import time

import numpy as np
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
import cpuinfo
import psutil

def get_cpu_info():
    """Returns CPU info using py-cpuinfo."""
    try:
        cpu_info = cpuinfo.get_cpu_info()
        return cpu_info['brand_raw']
    except Exception as e:
        return f"Error: {e}"

def get_gpu_info():
    """Returns GPU info using pynvml."""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
        gpu_name = nvmlDeviceGetName(handle)  # No need to decode
        return gpu_name
    except Exception as e:
        return "No CUDA enabled GPU found."

def get_ram_info():
    """Returns RAM info using psutil."""
    try:
        ram = psutil.virtual_memory()
        return f"Total: {ram.total / (1024 ** 3):.2f} GB, Available: {ram.available / (1024 ** 3):.2f} GB"
    except ImportError:
        return "psutil not installed. Cannot retrieve RAM info."

def write_result_header(file):
    file.write(f"# CPU Info: {get_cpu_info()}\n")
    file.write(f"# GPU Info: {get_gpu_info()}\n")
    file.write("# Run Number, Timestamp, Time (s)\n")

def generate_matrices(size):
    """Generates fresh matrices for each run."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return A, B

def get_formatted_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    return formatted_time