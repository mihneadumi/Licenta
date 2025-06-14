import customtkinter as ctk
import threading
import queue
import os
import traceback
import shutil
from PIL import Image

from constants.params import RUNS, SMALL_MATRIX_SIZE, MID_MATRIX_SIZE, BIG_MATRIX_SIZE, BIG_ARRAY_LENGTH, \
    MID_ARRAY_LENGTH, SMALL_ARRAY_LENGTH
from performance_profiling.kmeans_clustering.profile_kmeans_all import profile_and_save_stats
from performance_profiling.matrix_multiplication.profile_matrix_mult_all import run_matrix_multiplication_benchmark
from performance_profiling.radix_sort.profile_radix_sort_all import run_radix_sort_benchmark
from plotter import generate_all_plots
from utils.utils import get_cpu_info, get_ram_info, get_gpu_info


class BenchmarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Benchmark Runner")
        self.geometry("1200x700")
        self.log_queue = queue.Queue()
        self.is_running = False

        # --- grid layout (2x1) ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame (Controls) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        # --- Main Content Frame (Log and Plots) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # --- Sidebar Widgets ---
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Benchmark Controls",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Algorithm Selection
        self.algo_label = ctk.CTkLabel(self.sidebar_frame, text="Algorithm:", anchor="w")
        self.algo_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.algo_menu = ctk.CTkOptionMenu(self.sidebar_frame,
                                           values=["Matrix Multiplication", "Radix Sort", "K-Means Clustering"])
        self.algo_menu.grid(row=2, column=0, padx=20, pady=10)

        # Size Selection
        self.size_label = ctk.CTkLabel(self.sidebar_frame, text="Dataset Size:", anchor="w")
        self.size_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.size_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Small", "Medium", "Big", "Extra Large"])
        self.size_menu.grid(row=4, column=0, padx=20, pady=10)

        # Options
        self.options_label = ctk.CTkLabel(self.sidebar_frame, text="Options:", anchor="w")
        self.options_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.run_single_thread_var = ctk.BooleanVar(value=True)
        self.run_single_thread_check = ctk.CTkCheckBox(self.sidebar_frame, text="Run Single-Threaded",
                                                       variable=self.run_single_thread_var)
        self.run_single_thread_check.grid(row=6, column=0, padx=20, pady=10, sticky="w")
        self.run_gpu_var = ctk.BooleanVar(value=True)
        self.run_gpu_check = ctk.CTkCheckBox(self.sidebar_frame, text="Run GPU", variable=self.run_gpu_var)
        self.run_gpu_check.grid(row=7, column=0, padx=20, pady=10, sticky="w")

        # Action Buttons
        self.run_button = ctk.CTkButton(self.sidebar_frame, text="Run Benchmark", command=self.run_benchmark)
        self.run_button.grid(row=8, column=0, padx=20, pady=10, sticky="s")

        self.plot_button = ctk.CTkButton(self.sidebar_frame, text="Generate Plots",
                                         command=self.generate_and_load_plots)
        self.plot_button.grid(row=9, column=0, padx=20, pady=10)

        self.clear_log_button = ctk.CTkButton(self.sidebar_frame, text="Clear Log",
                                              command=lambda: self.log_textbox.delete("1.0", "end"))
        self.clear_log_button.grid(row=10, column=0, padx=20, pady=10)

        self.clear_results_button = ctk.CTkButton(self.sidebar_frame, text="Clear All Results",
                                                  command=self.clear_all_results,
                                                  fg_color="#D32F2F", hover_color="#B71C1C")
        self.clear_results_button.grid(row=11, column=0, padx=20, pady=20)

        # --- Main Frame Widgets ---
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)
        self.main_frame.grid_rowconfigure(2, weight=2)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Log Textbox
        self.log_textbox = ctk.CTkTextbox(self.main_frame, width=250)
        self.log_textbox.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Plot Controls
        self.plot_label = ctk.CTkLabel(self.main_frame, text="Select Plot to View:", anchor="w")
        self.plot_label.grid(row=1, column=0, padx=(10, 0), pady=10, sticky="w")
        self.plot_menu = ctk.CTkOptionMenu(self.main_frame, values=["No plots generated yet"],
                                           command=self.display_selected_plot)
        self.plot_menu.grid(row=1, column=1, padx=(0, 10), pady=10, sticky="e")

        # Plot Display Area
        self.plot_image_label = ctk.CTkLabel(self.main_frame, text="Generated plots will be displayed here.")
        self.plot_image_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.after(100, self.process_log_queue)
        self.log_system_info()

    def log_system_info(self):
        self.log_message("--- System Information ---")
        try:
            cpu_info = get_cpu_info()
            ram_info = get_ram_info()
            gpu_info = get_gpu_info()

            self.log_message(f"CPU: {cpu_info}")
            self.log_message(f"RAM: {ram_info}")
            self.log_message(f"GPU: {gpu_info}")
            self.log_message("--------------------------\n")

        except Exception as e:
            self.log_message(f"Error gathering system info: {e}")
            self.log_message("--------------------------\n")

    def log_message(self, message):
        self.log_queue.put(message)

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                self.log_textbox.insert("end", message + "\n")
                self.log_textbox.see("end")
                self.update_idletasks()
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_log_queue)

    def run_benchmark(self):
        if self.is_running:
            self.log_message("A benchmark is already in progress. Please wait.")
            return

        self.is_running = True
        self.run_button.configure(state="disabled", text="Running...")

        config = {
            "algorithm": self.algo_menu.get(),
            "size": self.size_menu.get(),
            "runs": RUNS,
            "run_single_thread": self.run_single_thread_var.get(),
            "run_gpu": self.run_gpu_var.get(),
        }

        thread = threading.Thread(target=self.benchmark_thread_worker, args=(config,))
        thread.daemon = True
        thread.start()

    def benchmark_thread_worker(self, config):
        self.log_message(f"--- Starting Benchmark: {config['algorithm']} ({config['size']}) ---")

        size_map_mm = {"Small": SMALL_MATRIX_SIZE, "Medium": MID_MATRIX_SIZE, "Big": BIG_MATRIX_SIZE,
                       "Extra Large": 10000}
        size_map_rs = {"Small": SMALL_ARRAY_LENGTH, "Medium": MID_ARRAY_LENGTH, "Big": BIG_ARRAY_LENGTH,
                       "Extra Large": BIG_ARRAY_LENGTH * 10}

        try:
            if config["algorithm"] == "Matrix Multiplication":
                size = size_map_mm.get(config["size"])
                self.log_message(f"Running Matrix Multiplication for size {size}x{size}...")
                run_matrix_multiplication_benchmark(
                    size=size,
                    runs=config["runs"],
                    run_single_thread=config["run_single_thread"],
                    run_gpu=config["run_gpu"]
                )

            elif config["algorithm"] == "Radix Sort":
                size = size_map_rs.get(config["size"])
                self.log_message(f"Running Radix Sort for array size {size:,}...")
                run_radix_sort_benchmark(
                    size=size,
                    runs=config["runs"],
                    run_single_thread=config["run_single_thread"],
                    run_gpu=config["run_gpu"]
                )

            elif config["algorithm"] == "K-Means Clustering":
                kmeans_params = {
                    "Small": (100000, 32, 10),
                    "Medium": (1000000, 32, 10),
                    "Big": (10000000, 32, 10),
                    "Extra Large": (20000000, 32, 10)
                }
                n_points, n_dims, k_clusters = kmeans_params.get(config["size"])
                self.log_message(f"Running K-Means for {n_points:,} points...")
                profile_and_save_stats(
                    n_points=n_points, n_dims=n_dims, n_clusters=k_clusters,
                    max_iters=50, tol=1e-4, total_runs=config["runs"],
                    run_single_thread_impl=config["run_single_thread"]
                )

            self.log_message("\n--- Benchmark Finished Successfully! ---")
            self.log_message("You can now generate and view plots.")
        except Exception as e:
            self.log_message(f"\n--- ERROR DURING BENCHMARK ---")
            self.log_message(f"Error: {e}")
            self.log_message(traceback.format_exc())
        finally:
            self.is_running = False
            self.run_button.configure(state="normal", text="Run Benchmark")

    def generate_and_load_plots(self):
        self.log_message("\n--- Generating Plots ---")
        self.plot_button.configure(state="disabled", text="Plotting...")
        self.update()

        try:
            generate_all_plots()
            self.log_message("Plot generation complete.")

            plot_dir = "visualizations_and_stats"
            found_plots = []
            for root, _, files in os.walk(plot_dir):
                for file in files:
                    if file.endswith(".png"):
                        relative_path = os.path.relpath(os.path.join(root, file), plot_dir)
                        found_plots.append(relative_path)

            if found_plots:
                self.log_message(f"Found {len(found_plots)} plots. Select one from the dropdown above.")
                self.plot_menu.configure(values=found_plots)
                self.plot_menu.set(found_plots[0])
                self.display_selected_plot(found_plots[0])
            else:
                self.log_message("No .png plot files found in 'visualizations_and_stats'.")
                self.plot_menu.configure(values=["No plots generated yet"])

        except Exception as e:
            self.log_message(f"\n--- ERROR DURING PLOTTING ---")
            self.log_message(f"Error: {e}")
            self.log_message(traceback.format_exc())
        finally:
            self.plot_button.configure(state="normal", text="Generate Plots")

    def clear_all_results(self):
        if self.is_running:
            self.log_message("Cannot clear results while a benchmark is in progress.")
            return

        self.log_message("\n--- Clearing all results and plots... ---")

        results_dir = 'results'
        plots_dir = 'visualizations_and_stats'

        blank_image = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        ctk_blank_image = ctk.CTkImage(light_image=blank_image, dark_image=blank_image, size=(1, 1))

        self.plot_image_label.configure(image=ctk_blank_image, text="Generated plots will be displayed here.")
        self.plot_image_label.image = ctk_blank_image # smartass gc
        self.plot_menu.configure(values=["No plots generated yet"])
        self.plot_menu.set("No plots generated yet")
        self.log_message("Cleared plot display.")

        try:
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
                os.makedirs(results_dir)
                self.log_message(f"Successfully cleared '{results_dir}' directory.")
            else:
                self.log_message(f"Directory '{results_dir}' does not exist. Nothing to clear.")

            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
                os.makedirs(plots_dir)
                self.log_message(f"Successfully cleared '{plots_dir}' directory.")
            else:
                self.log_message(f"Directory '{plots_dir}' does not exist. Nothing to clear.")

            self.log_message("--- Clearing process complete. ---")

        except Exception as e:
            self.log_message(f"An error occurred while clearing results: {e}")
            self.log_message(traceback.format_exc())

    def display_selected_plot(self, plot_path):
        full_path = os.path.join("visualizations_and_stats", plot_path)
        if not os.path.exists(full_path):
            self.plot_image_label.configure(text=f"Error: Plot not found at\n{full_path}")
            return

        try:
            image = Image.open(full_path)
            frame_w = self.main_frame.winfo_width()
            frame_h = self.plot_image_label.winfo_height()

            img_w, img_h = image.size
            ratio = min(frame_w / img_w, frame_h / img_h)

            if ratio < 0.95:
                new_w = int(img_w * ratio * 0.95)
                new_h = int(img_h * ratio * 0.95)
                image = image.resize((new_w, new_h), Image.LANCZOS)

            ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)

            self.plot_image_label.configure(image=ctk_image, text="")
            self.plot_image_label.image = ctk_image

        except Exception as e:
            self.plot_image_label.configure(text=f"Error loading plot: {e}")
            self.log_message(f"Error loading plot '{plot_path}': {e}")


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = BenchmarkApp()
    app.mainloop()
