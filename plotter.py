import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
from collections import defaultdict

# --- Configuration ---
RESULTS_DIR = 'results'
OUTPUT_DIR = 'visualizations_and_stats'
SUMMARY_STATS_FILE = os.path.join(OUTPUT_DIR, 'summary_statistics_excl_warmup.csv')
# Plotting Aesthetics
FIG_WIDTH = 6
FIG_DPI = 150
COMP_FIG_HEIGHT = 6
RUN_FIG_HEIGHT = 6
COMP_BAR_WIDTH = 0.5
LABEL_FONT_SIZE = 13
TITLE_FONT_SIZE = 15
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 12
BAR_LABEL_Y_FACTOR = 1.15
# --- End Configuration ---

# --- Helper Functions ---
def sanitize_filename(name):
    """Removes potentially problematic characters for filenames."""
    name = re.sub(r'[\\/*?:"<>|]+', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)
    return name


def get_implementation_sort_key(impl_name):
    """
    Determines the sort order for implementations in comparison plots.
    Order: 0. Single-threaded, 1. Multi-threaded, 2. GPU.
    Within each category, sorts alphabetically by original name.
    """
    name_lower = impl_name.lower()
    if 'gpu' in name_lower:
        return (2, impl_name)
    elif any(keyword in name_lower for keyword in ['multi_thread', 'multi-thread', 'parallel', 'numba']):
        if 'numba' in name_lower and ('single_thread' in name_lower or 'sequential' in name_lower):
            return (0, impl_name)
        return (1, impl_name)
    elif any(keyword in name_lower for keyword in ['single_thread', 'single-thread', 'sequential']):
        return (0, impl_name)
    else:
        return (0, impl_name)


def load_data_file(file_path):
    """Loads data from a single benchmark file."""
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        if 'Run' not in df.columns or 'Time(s)' not in df.columns:
            print(f"Warning: Required columns ('Run', 'Time(s)') not found in {file_path}. Skipping.")
            return None
        df['Time(s)'] = pd.to_numeric(df['Time(s)'], errors='coerce')
        df.dropna(subset=['Time(s)'], inplace=True)

        perf_col = None
        if 'GFLOPS' in df.columns:
            perf_col = 'GFLOPS'
        elif 'MElements/s' in df.columns:
            perf_col = 'MElements/s'
        elif 'PointsPerSec' in df.columns:
            perf_col = 'PointsPerSec'

        if perf_col:
            df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
            if (df[perf_col] <= 0).any():
                df = df[df[perf_col] > 0]
            df.dropna(subset=[perf_col], inplace=True)
            df['Performance_Metric'] = perf_col
        else:
            df['Performance_Metric'] = None

        if df.empty:
            return None
        return df
    except FileNotFoundError:
        print(f"Error: File not found {file_path}");
        return None
    except pd.errors.EmptyDataError:
        print(f"Warning: Skipping empty file {file_path}");
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}");
        return None

def plot_individual_run(df, algorithm, size_str, implementation, output_dir):
    """Generates and saves plots for a single benchmark result, excluding Run 1."""
    if df.empty: return

    sanitized_impl = sanitize_filename(implementation)
    base_filename = f"{sanitized_impl}"
    df_plot = df[df['Run'] > 1].copy()
    if df_plot.empty: return

    fig_width = FIG_WIDTH
    fig_height_run = RUN_FIG_HEIGHT

    # Time Plot
    plt.figure(figsize=(fig_width, fig_height_run))
    metric_label_time = 'Time (s)'
    metric_col_time = 'Time(s)'
    time_median = df_plot[metric_col_time].median()
    time_stdev = df_plot[metric_col_time].std() if len(df_plot[metric_col_time].dropna()) >= 2 else 0.0

    plt.plot(df_plot['Run'], df_plot[metric_col_time], marker='o', linestyle='-', label=metric_label_time)
    if pd.notna(time_median):
        plt.axhline(time_median, color='r', linestyle='--', linewidth=1.5, label=f'Median: {time_median:.4f}')
    if pd.notna(time_stdev) and time_stdev > 1e-9:
        plt.text(0.98, 0.95, f'Std Dev: {time_stdev:.4f}', transform=plt.gca().transAxes,
                 fontsize=ANNOTATION_FONT_SIZE, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    plt.xlabel('Run Number (Warm-up Excluded)', fontsize=LABEL_FONT_SIZE)
    plt.ylabel(metric_label_time, fontsize=LABEL_FONT_SIZE)
    title_str = f'Execution Time per Run \nAlg: {algorithm}\nConfig: {size_str}\nImpl: {implementation}'
    plt.title(title_str, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    time_plot_path = os.path.join(output_dir, f"{base_filename}_time_vs_run_excl_warmup.png")
    try:
        plt.savefig(time_plot_path, dpi=FIG_DPI)
    except Exception as e:
        print(f"Error saving plot {time_plot_path}: {e}")
    plt.close()

    # Performance Plot
    perf_col_name_series = df['Performance_Metric']
    perf_col = perf_col_name_series.iloc[0] if not perf_col_name_series.empty and pd.notna(
        perf_col_name_series.iloc[0]) else None

    if perf_col and perf_col in df_plot.columns and not df_plot[perf_col].empty:
        plt.figure(figsize=(fig_width, fig_height_run))
        metric_label_perf = perf_col
        perf_median = df_plot[perf_col].median()
        perf_stdev = df_plot[perf_col].std() if len(df_plot[perf_col].dropna()) >= 2 else 0.0

        plt.plot(df_plot['Run'], df_plot[perf_col], marker='x', linestyle='--', color='green', label=metric_label_perf)

        if pd.notna(perf_median):
            format_str = '{:,.0f}' if (
                        perf_col == 'PointsPerSec' and perf_median > 1000) else '{:,.2f}' if perf_median >= 1 else '{:.4f}'
            plt.axhline(perf_median, color='r', linestyle='--', linewidth=1.5,
                        label=f'Median: {format_str.format(perf_median)}')
        if pd.notna(perf_stdev) and perf_stdev > 1e-9:
            format_str_std = '{:,.0f}' if (
                        perf_col == 'PointsPerSec' and perf_stdev > 1000) else '{:,.2f}' if perf_stdev >= 1 else '{:.4f}'
            plt.text(0.98, 0.95, f'Std Dev: {format_str_std.format(perf_stdev)}', transform=plt.gca().transAxes,
                     fontsize=ANNOTATION_FONT_SIZE, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        plt.xlabel('Run Number (Warm-up Excluded)', fontsize=LABEL_FONT_SIZE)
        plt.ylabel(metric_label_perf, fontsize=LABEL_FONT_SIZE)
        title_str_perf = f'{perf_col} per Run\nAlg: {algorithm}\nConfig: {size_str}\nImpl: {implementation}'
        plt.title(title_str_perf, fontsize=TITLE_FONT_SIZE)
        plt.xticks(fontsize=TICK_FONT_SIZE)
        plt.yticks(fontsize=TICK_FONT_SIZE)
        if perf_col == 'PointsPerSec' and (df_plot[perf_col].max() > 1e5 if not df_plot[perf_col].empty else False):
            plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.tight_layout()
        perf_plot_path = os.path.join(output_dir,
                                      f"{base_filename}_{sanitize_filename(perf_col)}_vs_run_excl_warmup.png")
        try:
            plt.savefig(perf_plot_path, dpi=FIG_DPI)
        except Exception as e:
            print(f"Error saving plot {perf_plot_path}: {e}")
        plt.close()


def calculate_stats_excluding_warmup(series):
    """Calculates stats for a series already excluding the warm-up run."""
    if not isinstance(series, pd.Series):
        return {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': 0}
    data = series
    if data.empty or data.isnull().all():
        return {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': len(data)}
    mean = data.mean()
    median = data.median()
    valid_data = data.dropna()
    stdev = valid_data.std() if len(valid_data) >= 2 else 0.0
    return {'mean': mean, 'median': median, 'stdev': stdev, 'count': len(data)}


def plot_comparison(stats_dict, metric_name, unit, algorithm, size_str, output_dir, use_median=False):
    """Generates comparison bar plots with sorted bars, renamed single-thread impls, and dynamic Y-limit."""
    if not stats_dict: return

    implementations_orig_keys = list(stats_dict.keys())
    sorted_impl_keys = sorted(implementations_orig_keys, key=get_implementation_sort_key)

    stat_key = 'median' if use_median else 'mean'
    plot_title_stat = 'Median' if use_median else 'Average'
    sanitized_metric_name = sanitize_filename(metric_name)

    values = [stats_dict[impl_key].get(metric_name, {}).get(stat_key, float('nan')) for impl_key in sorted_impl_keys]
    values = [v if pd.notna(v) and v > 0 else float('nan') for v in values]
    errors = [stats_dict[impl_key].get(metric_name, {}).get('stdev', float('nan')) for impl_key in sorted_impl_keys]

    valid_indices = [i for i, v in enumerate(values) if pd.notna(v)]
    if not valid_indices: return

    filtered_impl_keys = [sorted_impl_keys[i] for i in valid_indices]
    filtered_values = [values[i] for i in valid_indices]
    filtered_errors = [errors[i] if pd.notna(errors[i]) else 0 for i in valid_indices]

    display_labels_for_plot = []
    for impl_key in filtered_impl_keys:
        sort_category, _ = get_implementation_sort_key(impl_key)
        if sort_category == 0:
            display_labels_for_plot.append('single_threaded_stats')
        else:
            display_labels_for_plot.append(impl_key)

    plt.figure(figsize=(FIG_WIDTH, COMP_FIG_HEIGHT))
    ax = plt.gca()

    x_positions = range(len(filtered_values))

    bars = ax.bar(x_positions, filtered_values, yerr=filtered_errors, capsize=5, color='skyblue', edgecolor='black',
                  log=True,
                  width=COMP_BAR_WIDTH)

    ax.set_ylabel(f'{plot_title_stat} {metric_name} ({unit})', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'Comparison of {plot_title_stat} {metric_name}\nAlg: {algorithm}\nConfig: {size_str}',
                 fontsize=TITLE_FONT_SIZE)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels_for_plot, rotation=30, ha='right', fontsize=TICK_FONT_SIZE)

    plt.yticks(fontsize=TICK_FONT_SIZE)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.7)
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)

    max_value_plotted = 0
    max_text_y_calculated = 0
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        if yval > max_value_plotted: max_value_plotted = yval

        if abs(yval) >= 1000 or (metric_name == 'PointsPerSec' and abs(yval) >= 1):
            fmt = '{:,.0f}'
        elif abs(yval) >= 1:
            fmt = '{:,.2f}'
        else:
            fmt = '{:.4f}'
        try:
            label_text = fmt.format(yval)
        except (ValueError, TypeError):
            label_text = f'{yval:.3g}'

        text_y_position = yval * BAR_LABEL_Y_FACTOR
        if text_y_position > max_text_y_calculated:
            max_text_y_calculated = text_y_position

        ax.text(x=bar.get_x() + bar.get_width() / 2.0,
                y=text_y_position,
                s=label_text, va='bottom', ha='center', fontsize=ANNOTATION_FONT_SIZE)

    plt.tight_layout()

    if max_value_plotted > 0:
        current_bottom_lim, current_top_lim = ax.get_ylim()
        if max_text_y_calculated >= current_top_lim * 0.85:
            new_top_lim = max_text_y_calculated * 1.25
            ax.set_ylim(bottom=current_bottom_lim, top=new_top_lim)
            plt.tight_layout()

    comp_plot_path = os.path.join(output_dir,
                                  f"comparison_{plot_title_stat.lower()}_{sanitized_metric_name}_excl_warmup_log.png")
    try:
        plt.savefig(comp_plot_path, dpi=FIG_DPI)
    except Exception as e:
        print(f"Error saving log comparison plot {comp_plot_path}: {e}")
    plt.close()


def generate_all_plots():
    """
    Main function to orchestrate the entire data loading, processing, and plotting workflow.
    This can be called from an external script, like a GUI application.
    """
    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input directory: {os.path.abspath(RESULTS_DIR)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"NOTE: All statistics and plots will exclude the first run (warm-up).")
    print(f"NOTE: Comparison plots will use a logarithmic Y-axis.")
    print(f"NOTE: Bar labels will use full numbers with thousand separators (no scientific notation).")
    print(
        f"NOTE: Plot fonts adjusted, comparison plot bars slimmed ({COMP_BAR_WIDTH}), width standardized ({FIG_WIDTH}in).")
    print(
        f"NOTE: Comparison plot bars will be ordered: single-thread, multi-thread, GPU. Single-threaded implementations will be labeled as 'single_threaded_stats'.")

    # --- Data Collection ---
    all_data = defaultdict(lambda: defaultdict(dict)) # Re-initialize for each call
    print("\n--- Starting Data Collection ---")
    for root, dirs, files in os.walk(RESULTS_DIR):
        if os.path.abspath(root).startswith(os.path.abspath(OUTPUT_DIR)):
            continue
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                try:
                    relative_path = os.path.relpath(file_path, RESULTS_DIR)
                    parts = relative_path.split(os.sep)
                    if len(parts) == 3:
                        algorithm, size_str, impl_file = parts
                        implementation_name = impl_file.replace('.txt', '')
                        df = load_data_file(file_path)
                        if df is not None and not df.empty:
                            all_data[algorithm][size_str][implementation_name] = df
                            algo_output_dir = os.path.join(OUTPUT_DIR, sanitize_filename(algorithm))
                            size_output_dir_actual = os.path.join(algo_output_dir, sanitize_filename(size_str))
                            os.makedirs(size_output_dir_actual, exist_ok=True)
                    else:
                        pass
                except Exception as e:
                    print(f"Error parsing path structure or loading data for {file_path}: {e}")
    print("--- Data Collection Finished ---")

    # --- Individual Run Plotting ---
    print("\n--- Generating Individual Run Plots (Excluding Warm-up) ---")
    for algorithm, sizes_dict in all_data.items():
        for size_str_key, implementations_dict in sizes_dict.items():
            for implementation_name, df_data in implementations_dict.items():
                algo_output_dir = os.path.join(OUTPUT_DIR, sanitize_filename(algorithm))
                size_output_dir_actual = os.path.join(algo_output_dir, sanitize_filename(size_str_key))
                plot_individual_run(df_data, algorithm, size_str_key, implementation_name, size_output_dir_actual)
    print("--- Individual Run Plotting Finished ---")

    # --- Statistics and Comparison Plotting ---
    print("\n--- Calculating Statistics and Generating Comparison Plots (Excluding Warm-up Run, Log Scale) ---")
    summary_rows = []
    for algorithm, sizes_dict in all_data.items():
        for size_str_key, implementations_dict in sizes_dict.items():
            algo_output_dir = os.path.join(OUTPUT_DIR, sanitize_filename(algorithm))
            size_output_dir_actual = os.path.join(algo_output_dir, sanitize_filename(size_str_key))
            current_run_stats = {}
            perf_metric_name_group, perf_metric_unit_group = None, None

            for df_impl_check in implementations_dict.values():
                if df_impl_check.empty: continue
                perf_col_check_series = df_impl_check['Performance_Metric']
                if not perf_col_check_series.empty and pd.notna(perf_col_check_series.iloc[0]):
                    perf_metric_name_group = perf_col_check_series.iloc[0]
                    if 'GFLOPS' in perf_metric_name_group:
                        perf_metric_unit_group = 'GFLOPS'
                    elif 'MElements/s' in perf_metric_name_group:
                        perf_metric_unit_group = 'MElements/s'
                    elif 'PointsPerSec' in perf_metric_name_group:
                        perf_metric_unit_group = 'Points/s'
                    else:
                        perf_metric_unit_group = ''
                    break

            for implementation_name, df_impl_data in implementations_dict.items():
                if df_impl_data.empty: continue
                df_stats = df_impl_data[df_impl_data['Run'] > 1].copy()
                if df_stats.empty:
                    time_stats = {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': 0}
                    perf_stats = {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': 0}
                else:
                    time_stats = calculate_stats_excluding_warmup(df_stats['Time(s)'])
                    perf_stats = {'mean': float('nan'), 'median': float('nan'), 'stdev': float('nan'), 'count': 0}
                    actual_perf_col_in_df = None
                    if perf_metric_name_group and perf_metric_name_group in df_stats.columns:
                        actual_perf_col_in_df = perf_metric_name_group
                        perf_stats = calculate_stats_excluding_warmup(df_stats[actual_perf_col_in_df])

                current_run_stats[implementation_name] = {'Time(s)': time_stats}
                if perf_metric_name_group:
                    current_run_stats[implementation_name][perf_metric_name_group] = perf_stats

                n_points_csv, d_dims_csv, k_clusters_csv = "N/A", "N/A", "N/A"
                if algorithm.lower().replace("_", "") == 'kmeansclustering':
                    match = re.match(r"N(\d+k?)_D(\d+)_K(\d+)", size_str_key, re.IGNORECASE)
                    if match:
                        n_points_csv = match.group(1).upper()
                        d_dims_csv = match.group(2)
                        k_clusters_csv = match.group(3)
                else:
                    n_points_csv = size_str_key

                summary_rows.append({'Algorithm': algorithm, 'Size': size_str_key,
                                     'N_Points': n_points_csv, 'D_Dims': d_dims_csv, 'K_Clusters': k_clusters_csv,
                                     'Implementation': implementation_name, 'Metric': 'Time(s)',
                                     'Mean': time_stats['mean'], 'Median': time_stats['median'],
                                     'StdDev': time_stats['stdev'], 'Count': time_stats['count']})

                summary_perf_metric_name = actual_perf_col_in_df if actual_perf_col_in_df else (
                    perf_metric_name_group if perf_metric_name_group else 'N/A')
                summary_rows.append({'Algorithm': algorithm, 'Size': size_str_key,
                                     'N_Points': n_points_csv, 'D_Dims': d_dims_csv, 'K_Clusters': k_clusters_csv,
                                     'Implementation': implementation_name,
                                     'Metric': summary_perf_metric_name,
                                     'Mean': perf_stats['mean'], 'Median': perf_stats['median'],
                                     'StdDev': perf_stats['stdev'], 'Count': perf_stats['count']})

            if current_run_stats:
                plot_comparison(current_run_stats, 'Time(s)', 's', algorithm, size_str_key, size_output_dir_actual,
                                use_median=False)
                plot_comparison(current_run_stats, 'Time(s)', 's', algorithm, size_str_key, size_output_dir_actual,
                                use_median=True)
                if perf_metric_name_group and perf_metric_unit_group is not None:
                    plot_comparison(current_run_stats, perf_metric_name_group, perf_metric_unit_group, algorithm,
                                    size_str_key, size_output_dir_actual, use_median=False)
                    plot_comparison(current_run_stats, perf_metric_name_group, perf_metric_unit_group, algorithm,
                                    size_str_key, size_output_dir_actual, use_median=True)

            gpu_impl, cpu_baseline_impl = None, None
            baseline_prefs = [
                'cpu_single_thread', 'cpu_sequential', 'sequential',
                'optimised_cpu_single_thread', 'cpu_numba_single_thread',
                'cpu_numba_sequential', 'baseline_cpu_single_thread'
            ]
            fallback_single_thread_terms = ['single', 'sequential']
            broader_cpu_prefs = ['cpu_numba', 'cpu_optimised', 'cpu_baseline', 'baseline_cpu', 'cpu']

            implementations_available_names = list(current_run_stats.keys())
            for impl_n in implementations_available_names:
                if 'gpu' in impl_n.lower(): gpu_impl = impl_n; break

            if gpu_impl:
                for pref_base in baseline_prefs:
                    if pref_base in implementations_available_names and pref_base != gpu_impl:
                        cpu_baseline_impl = pref_base;
                        break
                    potential_matches = [name for name in implementations_available_names if
                                         pref_base in name.lower() and name != gpu_impl]
                    if potential_matches: cpu_baseline_impl = sorted(potential_matches)[0]; break

                if not cpu_baseline_impl:
                    for term in fallback_single_thread_terms:
                        potential_matches = [name for name in implementations_available_names if
                                             term in name.lower() and 'gpu' not in name.lower() and name != gpu_impl]
                        if potential_matches:
                            cat_0_matches = [m for m in potential_matches if get_implementation_sort_key(m)[0] == 0]
                            if cat_0_matches: cpu_baseline_impl = sorted(cat_0_matches)[0]; break
                            cpu_baseline_impl = sorted(potential_matches)[0];
                            break

                if not cpu_baseline_impl:
                    for pref_base in broader_cpu_prefs:
                        potential_matches = [name for name in implementations_available_names if
                                             pref_base in name.lower() and 'gpu' not in name.lower() and name != gpu_impl]
                        if potential_matches:
                            cat_0_matches = [m for m in potential_matches if get_implementation_sort_key(m)[0] == 0]
                            if cat_0_matches: cpu_baseline_impl = sorted(cat_0_matches)[0]; break
                            cpu_baseline_impl = sorted(potential_matches)[0];
                            break

                if not cpu_baseline_impl:
                    non_gpu_cpus = [name for name in implementations_available_names if 'gpu' not in name.lower()]
                    if non_gpu_cpus:
                        cat_0_cpus = [n for n in non_gpu_cpus if get_implementation_sort_key(n)[0] == 0]
                        if cat_0_cpus:
                            cpu_baseline_impl = sorted(cat_0_cpus)[0]
                        else:
                            cpu_baseline_impl = sorted(non_gpu_cpus)[0]

            if gpu_impl and cpu_baseline_impl and \
                    gpu_impl in current_run_stats and cpu_baseline_impl in current_run_stats:
                median_time_gpu_stats = current_run_stats[gpu_impl].get('Time(s)', {})
                median_time_cpu_stats = current_run_stats[cpu_baseline_impl].get('Time(s)', {})
                median_time_gpu = median_time_gpu_stats.get('median', float('nan'))
                median_time_cpu = median_time_cpu_stats.get('median', float('nan'))

                if pd.notna(median_time_gpu) and pd.notna(median_time_cpu) and median_time_cpu > 1e-9:
                    speedup = median_time_cpu / median_time_gpu if median_time_gpu > 1e-9 else float('inf')
                    perc_improvement = ((median_time_cpu - median_time_gpu) / median_time_cpu) * 100

                    n_points_speedup, d_dims_speedup, k_clusters_speedup = "N/A", "N/A", "N/A"
                    if algorithm.lower().replace("_", "") == 'kmeansclustering':
                        match_speedup = re.match(r"N(\d+k?)_D(\d+)_K(\d+)", size_str_key, re.IGNORECASE)
                        if match_speedup:
                            n_points_speedup = match_speedup.group(1).upper()
                            d_dims_speedup = match_speedup.group(2)
                            k_clusters_speedup = match_speedup.group(3)
                    else:
                        n_points_speedup = size_str_key

                    summary_rows.append({'Algorithm': algorithm, 'Size': size_str_key,
                                         'N_Points': n_points_speedup, 'D_Dims': d_dims_speedup,
                                         'K_Clusters': k_clusters_speedup,
                                         'Implementation': f'GPU Speedup vs {cpu_baseline_impl}',
                                         'Metric': 'Speedup Factor (Median Time)',
                                         'Mean': speedup, 'Median': speedup,
                                         'StdDev': float('nan'), 'Count': 1})
                    summary_rows.append({'Algorithm': algorithm, 'Size': size_str_key,
                                         'N_Points': n_points_speedup, 'D_Dims': d_dims_speedup,
                                         'K_Clusters': k_clusters_speedup,
                                         'Implementation': f'GPU Improvement vs {cpu_baseline_impl}',
                                         'Metric': '% Improvement (Median Time)',
                                         'Mean': perc_improvement, 'Median': perc_improvement,
                                         'StdDev': float('nan'), 'Count': 1})

    print("--- Statistics Calculation and Comparison Plotting Finished ---")

    # --- Save Summary ---
    print(f"\n--- Saving Summary Statistics (Excluding Warm-up Run) ---")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        cols_order = ['Algorithm', 'Size', 'N_Points', 'D_Dims', 'K_Clusters',
                      'Implementation', 'Metric', 'Mean', 'Median', 'StdDev', 'Count']
        for col in cols_order:
            if col not in summary_df.columns:
                summary_df[col] = pd.NA
        summary_df = summary_df[cols_order]

        try:
            summary_df.to_csv(SUMMARY_STATS_FILE, index=False, float_format='%.5f')
            print(f"Summary statistics saved to: {SUMMARY_STATS_FILE}")
        except Exception as e:
            print(f"Error saving summary statistics CSV: {e}")
    else:
        print("No summary statistics were generated.")

    print("\n--- Script Execution Complete ---")


# Run manually
if __name__ == "__main__":
    generate_all_plots()
