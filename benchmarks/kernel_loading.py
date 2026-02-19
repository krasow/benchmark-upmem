import os
import csv
import re
from .core import Benchmark, TESTS_DIR, DEFAULT_DPUS

class KernelLoadingBenchmark(Benchmark):
    """Measures dpu_load overhead for small vs large DPU binaries."""
    def __init__(self):
        super().__init__("kernel_loading", "./bin/run",
                         relative_dir="kernel-loading", label="baseline")

    def prepare(self, dpus, iterations, large=True):
        self.update_params_file({
            "dpu_number": dpus,
            "iterations": iterations,
            "large": "true" if large else "false",
        })


def run_sweep(args, registry_config):
    output_csv = args.csv_file
    verbose = args.verbose
    iterations = args.iterations

    bench = KernelLoadingBenchmark()
    dpus_list = args.dpus if args.dpus else DEFAULT_DPUS

    # --- Print Plan ---
    print(f"\n{'='*60}")
    print(f"KERNEL LOADING SWEEP")
    print(f"{'='*60}")
    print(f"DPUs: {dpus_list}")
    print(f"Iterations: {iterations}")
    print(f"Variants: small, large")
    print(f"{'='*60}\n")

    # Initialize CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["kernel_size", "dpus", "benchmark", "time", "iterations"])

    for nr_dpus in dpus_list:
        for kernel_size, use_large in [("small", False), ("large", True)]:
            print(f"--- kernel={kernel_size}, dpus={nr_dpus} ---")

            bench.prepare(nr_dpus, iterations, large=use_large)

            if not bench.compile(verbose):
                print(f"  Compile failed for kernel={kernel_size}, dpus={nr_dpus}")
                continue

            out = bench.run(verbose)
            time_val = bench.parse_time(out)

            if time_val is not None:
                print(f"  {kernel_size} @ {nr_dpus} DPUs: {time_val} ms")
                with open(output_csv, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([kernel_size, nr_dpus, "kernel_loading", time_val, iterations])
            else:
                print(f"  Failed to parse time for kernel={kernel_size}, dpus={nr_dpus}")
                if verbose and out:
                    print(out)


def plot_results(csv_path):
    """Generate a plot from kernel loading sweep results."""
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Matplotlib/pandas not found. pip install matplotlib pandas")
        return

    df = pd.read_csv(csv_path)
    if "kernel_size" not in df.columns:
        print("CSV does not look like a kernel_loading result.")
        return

    try:
        plt.style.use('ggplot')
    except:
        pass

    colors = {"small": "#348ABD", "large": "#E24A33"}
    markers = {"small": "o", "large": "s"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for kernel_size in sorted(df['kernel_size'].unique()):
        subset = df[df['kernel_size'] == kernel_size].sort_values('dpus')
        time_per_iter = subset['time'] / subset['iterations']
        ax.plot(subset['dpus'], time_per_iter,
                marker=markers.get(kernel_size, 'x'),
                color=colors.get(kernel_size, 'gray'),
                label=f"{kernel_size} kernel",
                linewidth=2, markersize=8)

    ax.set_xlabel("Number of DPUs", fontsize=12)
    ax.set_ylabel("Time per iteration (ms)", fontsize=12)
    ax.set_title("Kernel Loading Overhead: Small vs Large DPU Binary", fontsize=14)
    ax.set_xticks(sorted(df['dpus'].unique()))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.5)
    fig.tight_layout()

    plot_file = csv_path.replace(".csv", "_plot.png")
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.close(fig)


def register(registry):
    def add_args(parser):
        parser.add_argument("--kernel-loading", action="store_true",
                            help="Sweep kernel loading overhead (small vs large DPU binary)")

    registry.register("kernel_loading", run_sweep, add_args)
