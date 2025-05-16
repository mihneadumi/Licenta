import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict

RESULTS_DIR = 'results'
OUTPUT_DIR = 'visualizations_and_stats'
SUMMARY_STATS_FILE = os.path.join(OUTPUT_DIR, 'summary_statistics_excl_warmup.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input directory: {os.path.abspath(RESULTS_DIR)}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
print(f"NOTE: All statistics and plots will exclude the first run (warm-up).")
print(f"NOTE: Comparison plots will use a logarithmic Y-axis.")
print(f"NOTE: Bar labels will use full numbers with thousand separators (no scientific notation).")

def sanitize_filename(name):
    """Removes potentially problematic characters for filenames."""
    name = re.sub(r'[\\/*?:"<>|]+', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)
    return name

all_data = defaultdict(lambda: defaultdict(dict))

def load_data_file(file_path):
    """Loads data from a single benchmark file."""
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        if 'Run' not in df.columns or 'Time(s)' not in df.columns:
            print(f"Warning: Skipping {file_path}. Missing 'Run' or 'Time(s)' column.")
            return None
        df['Time(s)'] = pd.to_numeric(df['Time(s)'], errors='coerce')
        df.dropna(subset=['Time(s)'], inplace=True)

        perf_col = None
        if 'GFLOPS' in df.columns: perf_col = 'GFLOPS'
        elif 'MElements/s' in df.columns: perf_col = 'MElements/s'

        if perf_col:
            df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
            if (df[perf_col] <= 0).any():
                 print(f"Warning: Non-positive performance values found in {file_path} for {perf_col}. Rows with values <= 0 will be dropped.")
                 df = df[df[perf_col] > 0]
            df.dropna(subset=[perf_col], inplace=True)
            df['Performance_Metric'] = perf_col
        else:
             df['Performance_Metric'] = None

        if df.empty or len(df) == 0:
             print(f"Warning: No valid data left in {file_path} after cleaning/filtering.")
             return None

        return df
    except FileNotFoundError: print(f"Error: File not found {file_path}"); return None
    except pd.errors.EmptyDataError: print(f"Warning: Skipping empty file {file_path}"); return None
    except Exception as e: print(f"Error processing file {file_path}: {e}"); return None

print("\n--- Starting Data Collection ---")
for root, dirs, files in os.walk(RESULTS_DIR):
    if os.path.abspath(root).startswith(os.path.abspath(OUTPUT_DIR)): continue
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            try:
                relative_path = os.path.relpath(file_path, RESULTS_DIR)
                parts = relative_path.split(os.sep)
                if len(parts) == 3:
                    algorithm, size, impl_file = parts
                    implementation_name = impl_file.replace('.txt', '')
                    df = load_data_file(file_path)
                    if df is not None:
                        all_data[algorithm][size][implementation_name] = df
                        algo_output_dir = os.path.join(OUTPUT_DIR, sanitize_filename(algorithm))
                        size_output_dir = os.path.join(algo_output_dir, sanitize_filename(size))
                        os.makedirs(size_output_dir, exist_ok=True)
                else:
                    print(f"Skipping {file_path}: Path structure relative to '{RESULTS_DIR}' ('{relative_path}') does not match 'algorithm/size/file.txt'")
            except Exception as e:
                print(f"Error parsing path structure for {file_path}: {e}")
print("--- Data Collection Finished ---")


def plot_individual_run(df, algorithm, size, implementation, output_dir):
    """Generates and saves plots for a single benchmark result, excluding Run 1."""
    if df.empty: return

    sanitized_impl = sanitize_filename(implementation)
    base_filename = f"{sanitized_impl}"

    df_plot = df[df['Run'] > 1].copy()

    if df_plot.empty:
        print(f"Info: Skipping individual plots for {algorithm}/{size}/{implementation} - no data points left after excluding Run 1.")
        return

    plt.figure(figsize=(10, 5))
    metric_label = 'Time (s)'
    metric_col = 'Time(s)'

    time_median = df_plot[metric_col].median()
    time_stdev = df_plot[metric_col].std() if len(df_plot) >= 2 else 0.0

    plt.plot(df_plot['Run'], df_plot[metric_col], marker='o', linestyle='-', label=metric_label)

    if pd.notna(time_median):
        plt.axhline(time_median, color='r', linestyle='--', linewidth=1.5, label=f'Median: {time_median:.4f}')

    if pd.notna(time_stdev):
        plt.text(0.98, 0.95, f'Std Dev: {time_stdev:.4f}',
                 transform=plt.gca().transAxes, # Use axes coordinates
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    plt.xlabel('Run Number (Warm-up Excluded)')
    plt.ylabel(metric_label)
    plt.title(f'Execution Time per Run (Excluding Warm-up)\nAlgorithm: {algorithm}, Size: {size}, Implementation: {implementation}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    time_plot_path = os.path.join(output_dir, f"{base_filename}_time_vs_run_excl_warmup.png")
    try: plt.savefig(time_plot_path)
    except Exception as e: print(f"Error saving plot {time_plot_path}: {e}")
    plt.close()

    perf_col = df['Performance_Metric'].iloc[0] if 'Performance_Metric' in df.columns and not df['Performance_Metric'].empty and pd.notna(df['Performance_Metric'].iloc[0]) else None

    if perf_col and perf_col in df_plot.columns and not df_plot[perf_col].empty:
        plt.figure(figsize=(10, 5))
        metric_label = perf_col

        perf_median = df_plot[perf_col].median()
        perf_stdev = df_plot[perf_col].std() if len(df_plot) >= 2 else 0.0

        plt.plot(df_plot['Run'], df_plot[perf_col], marker='x', linestyle='--', color='green', label=metric_label)

        if pd.notna(perf_median):
             plt.axhline(perf_median, color='r', linestyle='--', linewidth=1.5, label=f'Median: {perf_median:,.2f}')

        if pd.notna(perf_stdev):
             plt.text(0.98, 0.95, f'Std Dev: {perf_stdev:,.2f}',
                      transform=plt.gca().transAxes,
                      fontsize=9, verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        plt.xlabel('Run Number (Warm-up Excluded)')
        plt.ylabel(metric_label)
        plt.title(f'{perf_col} per Run (Excluding Warm-up)\nAlgorithm: {algorithm}, Size: {size}, Implementation: {implementation}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        perf_plot_path = os.path.join(output_dir, f"{base_filename}_{sanitize_filename(perf_col)}_vs_run_excl_warmup.png")
        try: plt.savefig(perf_plot_path)
        except Exception as e: print(f"Error saving plot {perf_plot_path}: {e}")
        plt.close()


print("\n--- Generating Individual Run Plots (Excluding Warm-up) ---")
for algorithm, sizes in all_data.items():
    for size, implementations in sizes.items():
        for implementation, df in implementations.items():
            algo_output_dir = os.path.join(OUTPUT_DIR, sanitize_filename(algorithm))
            size_output_dir = os.path.join(algo_output_dir, sanitize_filename(size))
            plot_individual_run(df, algorithm, size, implementation, size_output_dir)
print("--- Individual Run Plotting Finished ---")


def calculate_stats_excluding_warmup(series):
    if not isinstance(series, pd.Series) or len(series) < 2:
        return {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': 0 if not isinstance(series, pd.Series) else len(series)}
    data = series.iloc[1:]
    if data.empty or data.isnull().all():
        return {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': len(data)}
    mean = data.mean()
    median = data.median()
    valid_data = data.dropna()
    stdev = valid_data.std() if len(valid_data) >= 2 else 0.0
    return {'mean': mean, 'median': median, 'stdev': stdev, 'count': len(data)}


def plot_comparison(stats_dict, metric_name, unit, algorithm, size, output_dir, use_median=False):
    implementations = list(stats_dict.keys())
    stat_key = 'median' if use_median else 'mean'
    plot_title_stat = 'Median' if use_median else 'Average'
    sanitized_metric_name = sanitize_filename(metric_name)
    values = [stats_dict[impl].get(metric_name, {}).get(stat_key, float('nan')) for impl in implementations]
    values = [v if v > 0 else float('nan') for v in values]
    errors = [stats_dict[impl].get(metric_name, {}).get('stdev', float('nan')) for impl in implementations]
    valid_indices = [i for i, v in enumerate(values) if pd.notna(v)]
    if not valid_indices:
        print(f"Skipping log comparison plot for {metric_name} ({algorithm}/{size}): No valid positive data after excluding warm-up.")
        return
    implementations = [implementations[i] for i in valid_indices]
    values = [values[i] for i in valid_indices]
    errors = [errors[i] if pd.notna(errors[i]) else 0 for i in valid_indices]
    plt.figure(figsize=(12, 7))
    bars = plt.bar(implementations, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black', log=True)
    plt.ylabel(f'{plot_title_stat} {metric_name} ({unit}) (Excl. Warm-up, Log Scale)')
    plt.title(f'Comparison of {plot_title_stat} {metric_name} (Excluding Warm-up Run, Log Scale)\nAlgorithm: {algorithm}, Size: {size}')
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, which='major', axis='y', linestyle='-', linewidth=0.7)
    plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)
    for bar in bars:
        yval = bar.get_height()
        if abs(yval) >= 1:
            if abs(yval - round(yval)) < 0.001: fmt = '{:,.0f}'
            else: fmt = '{:,.2f}'
        else: fmt = '{:.4f}'
        try: label_text = fmt.format(yval)
        except ValueError: label_text = f'{yval:.3g}'
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, label_text, va='bottom', ha='center', fontsize=9)
    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, f"comparison_{plot_title_stat.lower()}_{sanitized_metric_name}_excl_warmup_log.png")
    try: plt.savefig(comp_plot_path)
    except Exception as e: print(f"Error saving log comparison plot {comp_plot_path}: {e}")
    plt.close()


print("\n--- Calculating Statistics and Generating Comparison Plots (Excluding Warm-up Run, Log Scale) ---")
summary_rows = []
for algorithm, sizes in all_data.items():
    for size, implementations in sizes.items():
        print(f"\nProcessing Comparisons for: Algorithm='{algorithm}', Size='{size}'")
        algo_output_dir = os.path.join(OUTPUT_DIR, sanitize_filename(algorithm))
        size_output_dir = os.path.join(algo_output_dir, sanitize_filename(size))
        current_run_stats = {}
        perf_metric_name_group, perf_metric_unit_group = None, None
        for df in implementations.values():
             if df.empty: continue
             perf_col_check = df['Performance_Metric'].iloc[0] if 'Performance_Metric' in df.columns and not df['Performance_Metric'].empty and pd.notna(df['Performance_Metric'].iloc[0]) else None
             if perf_col_check:
                 if not perf_metric_name_group:
                     perf_metric_name_group = perf_col_check
                     perf_metric_unit_group = 'GFLOPS' if 'GFLOPS' in perf_metric_name_group else ('MElements/s' if 'MElements/s' in perf_metric_name_group else '')
                 elif perf_metric_name_group != perf_col_check:
                      print(f"  Warning: Inconsistent performance metrics found for {algorithm}/{size}. Expected '{perf_metric_name_group}', found '{perf_col_check}'. Comparisons use '{perf_metric_name_group}'.")
        for implementation, df in implementations.items():
            if df.empty: continue
            time_stats = calculate_stats_excluding_warmup(df['Time(s)'])
            perf_stats = {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': 0}
            actual_perf_col_in_df = None
            if perf_metric_name_group and perf_metric_name_group in df.columns:
                actual_perf_col_in_df = perf_metric_name_group
                perf_stats = calculate_stats_excluding_warmup(df[actual_perf_col_in_df])
            current_run_stats[implementation] = {'Time(s)': time_stats}
            if perf_metric_name_group: current_run_stats[implementation][perf_metric_name_group] = perf_stats
            summary_rows.append({'Algorithm': algorithm, 'Size': size, 'Implementation': implementation, 'Metric': 'Time(s)', 'Mean': time_stats['mean'], 'Median': time_stats['median'], 'StdDev': time_stats['stdev'], 'Count': time_stats['count']})
            summary_rows.append({'Algorithm': algorithm, 'Size': size, 'Implementation': implementation, 'Metric': actual_perf_col_in_df if actual_perf_col_in_df else (perf_metric_name_group if perf_metric_name_group else 'N/A'), 'Mean': perf_stats['mean'], 'Median': perf_stats['median'], 'StdDev': perf_stats['stdev'], 'Count': perf_stats['count']})
        if current_run_stats:
            plot_comparison(current_run_stats, 'Time(s)', 's', algorithm, size, size_output_dir, use_median=False)
            plot_comparison(current_run_stats, 'Time(s)', 's', algorithm, size, size_output_dir, use_median=True)
            if perf_metric_name_group:
                plot_comparison(current_run_stats, perf_metric_name_group, perf_metric_unit_group, algorithm, size, size_output_dir, use_median=False)
                plot_comparison(current_run_stats, perf_metric_name_group, perf_metric_unit_group, algorithm, size, size_output_dir, use_median=True)
        gpu_impl, cpu_baseline_impl = None, None
        baseline_prefs = ['cpu_single_thread_stats', 'optimised_cpu_single_thread_stats', 'cpu_numba_stats']
        implementations_available = list(current_run_stats.keys())
        for impl in implementations_available:
            if impl.lower().startswith('gpu') or 'gpu_' in impl.lower() or '_gpu' in impl.lower(): gpu_impl = impl; break
        for pref in baseline_prefs:
            if pref in implementations_available: cpu_baseline_impl = pref; break
        if not cpu_baseline_impl and gpu_impl:
            preferred_cpu_found = False; fallback_impl = None
            for impl in implementations_available:
                 if impl in baseline_prefs: cpu_baseline_impl = impl; preferred_cpu_found = True; break
            if not preferred_cpu_found:
                 for impl in implementations_available:
                    is_gpu = impl.lower().startswith('gpu') or 'gpu_' in impl.lower() or '_gpu' in impl.lower()
                    if not is_gpu:
                         if 'multi_thread' not in impl.lower(): cpu_baseline_impl = impl; break
                         elif fallback_impl is None: fallback_impl = impl
                 if not cpu_baseline_impl: cpu_baseline_impl = fallback_impl
        if gpu_impl and cpu_baseline_impl:
            median_time_gpu_stats = current_run_stats.get(gpu_impl, {}).get('Time(s)', {})
            median_time_cpu_stats = current_run_stats.get(cpu_baseline_impl, {}).get('Time(s)', {})
            median_time_gpu = median_time_gpu_stats.get('median', float('nan'))
            median_time_cpu = median_time_cpu_stats.get('median', float('nan'))
            if pd.notna(median_time_gpu) and pd.notna(median_time_cpu) and median_time_gpu > 1e-9:
                speedup = median_time_cpu / median_time_gpu
                perc_improvement = ((median_time_cpu - median_time_gpu) / median_time_cpu) * 100 if median_time_cpu > 1e-9 else float('inf')
                print(f"  GPU Speedup vs {cpu_baseline_impl} (Median Time, Excl. Warm-up): {speedup:.2f}x")
                summary_rows.append({'Algorithm': algorithm, 'Size': size, 'Implementation': f'GPU Speedup vs {cpu_baseline_impl}', 'Metric': 'Speedup Factor (Median Time)', 'Mean': speedup, 'Median': speedup, 'StdDev': float('nan'), 'Count': float('nan')})
                summary_rows.append({'Algorithm': algorithm, 'Size': size, 'Implementation': f'GPU Improvement vs {cpu_baseline_impl}', 'Metric': '% Improvement (Median Time)', 'Mean': perc_improvement, 'Median': perc_improvement, 'StdDev': float('nan'), 'Count': float('nan')})
            else:
                print(f"  Could not calculate speedup for {gpu_impl} vs {cpu_baseline_impl}: Missing/invalid median time data (GPU: {median_time_gpu}, CPU: {median_time_cpu}).")
        elif gpu_impl: print(f"  Found GPU ({gpu_impl}) but no suitable CPU baseline ({baseline_prefs} or other non-GPU) found for {algorithm}/{size} speedup calculation.")
        elif implementations_available: pass

print("--- Statistics Calculation and Comparison Plotting Finished ---")

print(f"\n--- Saving Summary Statistics (Excluding Warm-up Run) ---")
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[['Algorithm', 'Size', 'Implementation', 'Metric', 'Mean', 'Median', 'StdDev', 'Count']]
    try:
        summary_df.to_csv(SUMMARY_STATS_FILE, index=False, float_format='%.5f')
        print(f"Summary statistics saved to: {SUMMARY_STATS_FILE}")
    except Exception as e:
        print(f"Error saving summary statistics CSV: {e}")
else:
    print("No summary statistics were generated.")

print("\n--- Script Execution Complete ---")