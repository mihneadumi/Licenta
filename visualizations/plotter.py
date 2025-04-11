import pandas as pd
import matplotlib.pyplot as plt
import os

def read_data_from_file(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', engine='c', skipinitialspace=True)

        if df.shape[1] != 5:
            print(f"Skipping file {file_path} due to unexpected number of columns (found {df.shape[1]} columns).")
            return pd.DataFrame()

        df['Data Size'] = df['Data Size'].str.strip().str.replace(' MB', '', regex=False).astype(float)
        df['MB/min'] = df['MB/min'].str.replace(' MB/min', '', regex=False).astype(float)
        df['Time(s)'] = pd.to_numeric(df['Time(s)'], errors='coerce')

        return df

    except pd.errors.ParserError as e:
        print(f"Error parsing file {file_path}: {e}")
        return pd.DataFrame()

def generate_plots(df, size, algorithm):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Run'], df['Time(s)'], marker='o', color='b', label=f'Time (s) for {algorithm}')
    plt.xlabel('Run Number')
    plt.ylabel('Time (s)')
    plt.title(f'Execution Time per Run for {algorithm} (Size: {size})')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'plots/{algorithm}_{size}_time.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(df['Run'], df['MB/min'], color='g', label=f'Data Throughput for {algorithm}')
    plt.xlabel('Run Number')
    plt.ylabel('Data Throughput (MB/min)')
    plt.title(f'Data Throughput per Run for {algorithm} (Size: {size})')
    plt.ylim(df['MB/min'].min() - 10, df['MB/min'].max() + 10)

    average_value = df['MB/min'].mean()
    plt.axhline(average_value, color='b', linestyle='--', label=f'Average: {average_value:.2f} MB/min')

    plt.grid(True)
    plt.legend()
    plt.savefig(f'plots/{algorithm}_{size}_throughput.png')
    plt.close()

def process_results(directory):
    os.makedirs('plots', exist_ok=True)

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")

                # Get the grandparent folder name (this will be the algorithm type)
                grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

                if 'matrix_multiplication' in grandparent_folder:
                    algorithm = 'Matrix Multiplication'
                elif 'radix_sort' in grandparent_folder:
                    algorithm = 'Radix Sort'
                else:
                    print(f"Skipping file due to unknown folder structure: {grandparent_folder}")
                    continue

                match file_name:
                    case 'cpu_multi_thread_stats.txt':
                        algorithm += '_CPU_multi_threaded'
                    case 'cpu_single_thread_stats.txt':
                        algorithm += '_CPU_single_threaded'
                    case 'gpu_acceleration_stats.txt':
                        algorithm += '_GPU'
                    case 'optimised_cpu_single_thread_stats.txt':
                        algorithm += '_Optimised_CPU_single_threaded'

                size_dir = os.path.basename(os.path.dirname(file_path))

                df = read_data_from_file(file_path)

                if not df.empty:
                    generate_plots(df, size_dir, algorithm)

if __name__ == "__main__":
    results_dir = 'results'
    process_results(results_dir)
