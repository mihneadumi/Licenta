from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
import cpuinfo

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
        return f"Error: {e}"
