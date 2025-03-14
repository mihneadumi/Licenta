import os
import numpy as np

def generate_random_arrays(size, num_arrays=10, max_value=1000000):
    """Generates multiple random arrays for sorting benchmarking."""
    arrays = [np.random.randint(1, max_value, size) for _ in range(num_arrays)]
    return np.array(arrays)

def save_arrays(size, num_arrays=10):
    """Saves generated arrays to a .npz file."""
    output_dir = "performance_profiling/radix_sort/"
    os.makedirs(output_dir, exist_ok=True)

    arrays = generate_random_arrays(size, num_arrays)
    np.savez(f"{output_dir}/pre_generated_arrays.npz", arrays=arrays)

    print(f"Generated {num_arrays} random arrays of size {size}.")

def run_array_generation(size=1000):
    """Runs the array generation process."""
    save_arrays(size)

if __name__ == "__main__":
    run_array_generation()
