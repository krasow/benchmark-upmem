import os
import re
import subprocess
import csv
import sys

TESTS_DIR = "/scratch/david/benchmark-upmem/tests"
ENV_FILE = os.path.join(TESTS_DIR, ".localenv")
DEFAULT_DPUS = [64, 128, 256, 512, 1024]

class Benchmark:
    """Base class for a benchmark."""
    def __init__(self, name, exec_cmd, relative_dir=None, label=None, param_file="Param.h"):
        self.name = name
        self.dir = os.path.join(TESTS_DIR, relative_dir if relative_dir else name)
        self.exec_cmd = exec_cmd
        self.label = label if label else name
        self.param_file = os.path.join(self.dir, param_file)
    
    def update_params_file(self, replacements):
        """Updates the Parameter header file with given replacements keys/values."""
        if not os.path.exists(self.param_file):
            print("Warning: Param file {0} not found.".format(self.param_file))
            return

        with open(self.param_file, 'r') as f:
            content = f.read()

        for pattern, value in replacements.items():
            if pattern == "OPERATION":
                # Handle macro replacement: #define OPERATION(a, b) ...
                regex = r"(#define\s+OPERATION\s*\(\w+,\s*\w+\)\s*)(.*)"
                content = re.sub(regex, r"\1" + str(value), content)
            else:
                # Match assignments to const/uint32_t/uint64_t etc.
                regex = r"(\b{0}\s*=\s*)([^;]+);".format(pattern)
                content = re.sub(regex, r"\g<1>{0};".format(value), content)
        
        with open(self.param_file, 'w') as f:
            f.write(content)

    def compile(self, verbose=False, extra_flags=""):
        """Runs make clean && make with extra flags."""
        command = "make clean && make"
        if extra_flags:
            command = "EXTRA_FLAGS=\"{0}\" {1}".format(extra_flags, command)
        return self._run_shell(command, verbose)

    def run(self, verbose=False, env=None, **kwargs):
        """Runs the benchmark executable."""
        return self._run_shell(self.exec_cmd, verbose, env)

    def _run_shell(self, command, verbose=False, extra_env=None):
        """Executes a shell command with the necessary environment."""
        full_command = "source {0} && {1}".format(ENV_FILE, command)
        if verbose:
            print("[{0}] Executing: {1}".format(self.name, full_command))
        
        cwd = self.dir
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        try:
            result = subprocess.run(
                ["/bin/bash", "-c", full_command], 
                capture_output=True, 
                text=True, 
                cwd=cwd, 
                env=env
            )
            
            if verbose:
                if result.stdout:
                    print("[{0}] STDOUT:\n{1}".format(self.name, result.stdout))
                if result.stderr:
                    print("[{0}] STDERR:\n{1}".format(self.name, result.stderr))

            if result.returncode != 0:
                print("Error running command: {0}".format(command))
                if not verbose:
                    print(result.stderr)
                return None
            return result.stdout
        except Exception as e:
            print("Subprocess exception: {0}".format(e))
            return None

    def parse_time(self, stdout):
        """Parses execution time from stdout based on the label."""
        if not stdout:
            return None
        
        match = re.search(r"{0}\s*\(ms\):\s*([0-9.]+)".format(self.label), stdout)
            
        if match:
            return float(match.group(1))
        return None

    def parse_verification(self, stdout):
        """Parses internal correctness check from stdout."""
        if not stdout:
            return None
        
        # Success strings
        if "All results match" in stdout or "the result is correct" in stdout:
            return True
        
        # Failure strings
        if "Mismatch at index" in stdout or "result mismatch at position" in stdout:
            return False
            
        return None # No verification performed

class CPUBenchmark(Benchmark):
    def __init__(self, suite_name):
        super().__init__(name="cpu_baseline", exec_cmd="./generate_ref", relative_dir=f"cpu-verification/{suite_name}", label="cpu_baseline")
        self.suite_name = suite_name

    def prepare(self, *args, **kwargs):
        # Specific prepare logic can go here if needed
        pass

    def run(self, verbose=False):
        if not self.compile(verbose):
            return None
        return super().run(verbose)

class LibVectorDPU(Benchmark):
    def __init__(self, name="libvectordpu", exec_cmd="./run", relative_dir="libvectordpu", label="libvectordpu"):
        super().__init__(name, exec_cmd, relative_dir, label)

    def run(self, verbose=False, dpus=None):
        env = {"NR_DPUS": str(dpus)} if dpus else {}
        return super().run(verbose, env=env)

    def rebuild_library(self, use_pipeline, use_logging=False, use_trace=False, verbose=False):
        src_dir = os.path.join(os.path.dirname(TESTS_DIR), "opt", "vectordpu_src")
        dest_dir = os.path.join(os.path.dirname(TESTS_DIR), "opt", "vectordpu")
        
        pipeline_val = 1 if use_pipeline else 0
        logging_val = 1 if use_logging else 0
        trace_val = 1 if use_trace else 0
        
        command = f"make clean && DESTDIR={dest_dir} make install BACKEND=hw PIPELINE={pipeline_val} LOGGING={logging_val} TRACE={trace_val} CXX_STANDARD=c++17"
        
        if verbose:
            print(f"[libvectordpu] Rebuilding library with PIPELINE={pipeline_val}, LOGGING={logging_val}, TRACE={trace_val}...")
        
        full_command = f"source {ENV_FILE} && {command}"
        try:
            result = subprocess.run(["/bin/bash", "-c", full_command], capture_output=True, text=True, cwd=src_dir)
            if result.returncode != 0:
                print("Error rebuilding libvectordpu library:")
                print(result.stderr)
                return False
            if verbose: print(result.stdout)
            return True
        except Exception as e:
            print(f"Exception during library rebuild: {e}")
            return False

class SuiteRegistry:
    """Registry for benchmark suites."""
    def __init__(self):
        self.suites = {}

    def register(self, name, runner_func, add_args_func):
        self.suites[name] = {
            "runner": runner_func,
            "add_args": add_args_func
        }

    def get_suite(self, name):
        return self.suites.get(name)

    def list_suites(self):
        return list(self.suites.keys())

class Plotter:
    """Handles parsing CSV logs and generating plots/charts."""
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def plot(self):
        """Generates plots from the CSV data."""
        if not os.path.exists(self.csv_path):
            print("CSV file {0} not found.".format(self.csv_path))
            return

        try: 
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not found. Please install it: pip install matplotlib")
            return
        try:
            import pandas as pd
            import numpy as np
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print("Error reading CSV: {0}".format(e))
            return

        # Check required columns
        base_required_cols = {"operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling"}
        if not base_required_cols.issubset(df.columns):
            print("CSV missing required columns. Expected at least: {0}".format(base_required_cols))
            return

        # Create 'variant' column for differentiation
        def create_variant(row):
            label = row['benchmark']
            if "libvectordpu" in label:
                if 'pipeline' in row and row['pipeline'] == 1:
                    label += " [PIPELINE]"
                if 'logging' in row and row['logging'] == 1:
                    label += " [LOGGING]"
            return label
        
        df['variant'] = df.apply(create_variant, axis=1)

        # Deduplicate: for non-libvectordpu benchmarks the pipeline/logging flags
        # are meaningless, so multiple sweep runs create duplicate rows with
        # identical variant names but different times.  Average them out.
        group_keys = ['operation', 'elements_per_dpu', 'total_elements', 'dpus', 'variant', 'scaling']
        df = df.groupby(group_keys, as_index=False).agg({'time': 'mean', 'benchmark': 'first',
                                                          'pipeline': 'max', 'logging': 'max'})

        # Setup standard styles
        benchmark_colors = {
            "simplepim": "#E24A33",   # Reddish
            "libvectordpu": "#348ABD", # Blueish
            "baseline": "#00B050",     # Greenish
            "simplepim_linreg": "#E24A33",
            "libvectordpu_linreg": "#348ABD",
            "baseline_fused": "#00B050",        # Greenish (compiled fused)
            "baseline_interpreter": "#8B5CF6",  # Purple (interpreter-style)
        }
        
        # Markers for different elements per dpu
        markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D"]
        for scaling_type in df['scaling'].unique():
            scaling_df = df[df['scaling'] == scaling_type]
            
            # Determine grouping column
            if scaling_type == 'weak':
                group_col = 'elements_per_dpu'
            else:
                group_col = 'total_elements'

            # 1. Line Plots (Per Operation)
            for op in scaling_df['operation'].unique():
                plt.figure(figsize=(10, 6))
                op_df = scaling_df[scaling_df['operation'] == op]
                unique_groups = sorted(op_df[group_col].unique())
                group_markers = {val: markers[i % len(markers)] for i, val in enumerate(unique_groups)}

                # Iterate over variants instead of benchmarks
                for variant in op_df['variant'].unique():
                    bench_df = op_df[op_df['variant'] == variant]
                    base_bench = variant.split(" [")[0]
                    
                    for group_val in unique_groups:
                        subset = bench_df[bench_df[group_col] == group_val].sort_values('dpus')
                        if not subset.empty:
                            if scaling_type == 'weak':
                                label = "{0} ({1}/dpu)".format(variant, group_val)
                            else:
                                label = "{0} (Total: {1})".format(variant, group_val)
                            
                            plt.plot(subset['dpus'], subset['time'], 
                                     marker=group_markers.get(group_val, 'x'),
                                     color=benchmark_colors.get(base_bench, 'gray'),
                                     label=label,
                                     linestyle='--' if '[PIPELINE]' in variant else '-')

                plt.title("Benchmark Performance ({0} - {1} scaling)".format(op, scaling_type))
                plt.xlabel("Number of DPUs")
                plt.ylabel("Execution Time (ms)")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                
                plot_filename = "plot_{0}_{1}_scaling.png".format(op, scaling_type)
                plt.savefig(plot_filename)
                print("Plot saved to {0}".format(plot_filename))
                plt.close()

            # 2. Clustered Bar Charts (One per DPU count, log scale)
            try:
                try: 
                    plt.style.use('ggplot')
                except: 
                    pass

                pivot_df = scaling_df.pivot_table(index=['dpus', 'operation', group_col], 
                                                 columns='variant', 
                                                 values='time').reset_index()
                
                all_variants = sorted([c for c in pivot_df.columns if c not in ['dpus', 'operation', group_col] and c in df['variant'].unique()])
                dpus_list = sorted(pivot_df['dpus'].unique())
                all_ops = sorted(pivot_df['operation'].unique())
                all_elements = sorted(pivot_df[group_col].unique())

                for dpu_count in dpus_list:
                    dpu_subset = pivot_df[pivot_df['dpus'] == dpu_count]
                    if dpu_subset.empty: continue

                    fig, ax = plt.subplots(figsize=(max(14, len(all_elements) * 3), 7))
                    
                    added_labels = set()
                    x_tick_pos = []
                    x_tick_labels = []
                    cursor = 0.0  # running x position
                    element_gap = 1.5
                    op_gap = 0.4
                    bar_width = 0.35

                    for ei, element in enumerate(all_elements):
                        if ei > 0:
                            cursor += element_gap

                        # Which ops have data for this element?
                        elem_ops = [op for op in all_ops 
                                    if not dpu_subset[(dpu_subset[group_col] == element) & (dpu_subset['operation'] == op)].empty]
                        if not elem_ops:
                            continue

                        group_start = cursor
                        for oi, op in enumerate(elem_ops):
                            if oi > 0:
                                cursor += op_gap
                            
                            row = dpu_subset[(dpu_subset[group_col] == element) & (dpu_subset['operation'] == op)]
                            if row.empty: continue
                            
                            # Only place bars for variants that have data for this op
                            present = [v for v in all_variants if v in row.columns and not pd.isna(row[v].values[0])]
                            
                            op_start = cursor
                            for ki, variant in enumerate(present):
                                val = row[variant].values[0]
                                base_bench = variant.split(" [")[0]
                                color = benchmark_colors.get(base_bench, 'gray')
                                
                                hatch = ""
                                if "[PIPELINE]" in variant: hatch = "//"
                                if "[LOGGING]" in variant: hatch = ".."

                                lbl = variant if variant not in added_labels else ""
                                ax.bar(cursor, val, bar_width,
                                       color=color, hatch=hatch, edgecolor='black', alpha=0.85,
                                       label=lbl)
                                if lbl:
                                    added_labels.add(variant)
                                cursor += bar_width
                            
                            # Op label centered under its bars
                            op_center = (op_start + cursor - bar_width) / 2
                            ax.text(op_center, 0, op, ha='center', va='top', fontsize=8,
                                    fontweight='bold', transform=ax.get_xaxis_transform())

                        # Element label centered under this element group
                        group_center = (group_start + cursor - bar_width) / 2
                        x_tick_pos.append(group_center)
                        x_tick_labels.append(f"{element:,}")

                    ax.set_yscale('log')
                    ax.set_ylabel('Execution Time (ms, log scale)', fontsize=12)
                    ax.set_xlabel('Elements/DPU' if scaling_type == 'weak' else 'Number of Elements', fontsize=12, labelpad=28)
                    ax.set_title(f'Benchmark Comparison ({dpu_count} DPUs)', fontsize=14)
                    
                    ax.set_xticks(x_tick_pos)
                    ax.set_xticklabels(x_tick_labels, fontsize=11)
                    
                    if added_labels:
                        ax.legend(fontsize=9, loc='upper left')
                    ax.grid(True, axis='y', alpha=0.5)
                    fig.tight_layout()
                    plt.subplots_adjust(bottom=0.15)
                    
                    bar_filename = f"bar_plot_{scaling_type}_{dpu_count}_dpus.png"
                    plt.savefig(bar_filename)
                    print(f"Bar chart saved to {bar_filename}")
                    plt.close(fig)

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error generating advanced bar charts: {e}")

def execute_sweep(args, benchmarks, operations, metric_arg="param", output_csv="sweep_results.csv", cpu_benchmark=None, extra_cols=None, elements_per_dpu_list=None, total_elements_list=None):
    """
    Executes a parameter sweep across common dimensions (scaling, DPUs) and specific operations.
    
    Args:
        args: Parsed CLI arguments (must have dpus, scaling, verbose, check, warmup, iterations).
        benchmarks: List of benchmark objects to run.
        operations: List of tuples (op_name, op_param) to iterate over.
        metric_arg: Name of the specific argument passed to prepare() as the 3rd arg (e.g. "op_val" or "dim").
        output_csv: Path to output CSV.
        cpu_benchmark: Optional CPUBenchmark instance for baseline.
        extra_cols: Optional dict of {col_name: value} to add to CSV for every row (e.g. {"dim": 10}).
        elements_per_dpu_list: List of elements per DPU for weak scaling.
        total_elements_list: List of total elements for strong scaling.
    """
    import csv 
    
    verbose = args.verbose
    check = args.check
    warmup = args.warmup
    iterations = getattr(args, 'iterations', 1)

    # Defaults if not provided
    if elements_per_dpu_list is None:
        elements_per_dpu_list = [1024*1024, 2*1024*1024]
    if total_elements_list is None:
        total_elements_list = [128*1024*1024, 256*1024*1024]
    
    # 1. Initialize CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            cols = ["operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling", "pipeline", "logging"]
            if extra_cols:
                cols.extend(extra_cols.keys())
            writer.writerow(cols)

    # 2. Determine Sweep Configurations (Scaling & DPUs)
    # Default to DEFAULT_DPUS if not set, handled by sweep.py usually
    dpus_list = args.dpus if args.dpus else DEFAULT_DPUS
    
    scaling_modes = ["weak", "strong"] if args.scaling == "both" else [args.scaling]
    
    # Scaling defaults
    # elements_per_dpu_list and total_elements_list are now args
    
    # --- Print Benchmark Plan ---
    print(f"\n{'='*60}")
    print(f"BENCHMARK SWEEP CONFIGURATION")
    print(f"{'='*60}")
    print(f"Operations: {[op[0] for op in operations]}")
    print(f"Scaling Modes: {scaling_modes}")
    print(f"DPUs: {dpus_list}")
    if "weak" in scaling_modes:
        print(f"Weak Scaling Elements/DPU: {elements_per_dpu_list}")
    if "strong" in scaling_modes:
        print(f"Strong Scaling Total Elements: {total_elements_list}")
    print(f"Benchmarks: {[b.name for b in benchmarks]}")
    if extra_cols:
        print(f"Extra Config: {extra_cols}")
    if cpu_benchmark and check:
        print(f"CPU Verification: Enabled ({cpu_benchmark.name})")
    print(f"{'='*60}\n")
    # ----------------------------

    for op_name, op_param in operations:
        for mode in scaling_modes:
            sweep_configs = []
            if mode == "weak":
                for elems in elements_per_dpu_list:
                    sweep_configs.append({"elems_per_dpu": elems, "total": None})
            else: # strong
                for total in total_elements_list:
                    sweep_configs.append({"elems_per_dpu": None, "total": total})
            
            for config in sweep_configs:
                for nr_dpus in dpus_list:
                    if mode == "weak":
                        elems_per_dpu = config["elems_per_dpu"]
                        nr_elements = nr_dpus * elems_per_dpu
                        scaling_label = "weak"
                    else:
                        nr_elements = config["total"]
                        elems_per_dpu = nr_elements // nr_dpus
                        scaling_label = "strong"
                    
                    results = {}
                    
                    # 3. CPU Baseline (Optional)
                    if check and cpu_benchmark:
                        print(f"\n--- Generating CPU Baseline for {op_name} (N={nr_elements}) ---")
                        # This prepare call assumes a specific signature. 
                        # To be safe, we pass op_param as 3rd arg.
                        cpu_benchmark.prepare(nr_dpus, nr_elements, op_param, warmup, iterations, check=check, load_ref=True)
                        cpu_out = cpu_benchmark.run(verbose)
                        cpu_time = cpu_benchmark.parse_time(cpu_out)
                        if cpu_time: results["cpu_baseline"] = cpu_time

                    print(f"\n--- Sweeping: op={op_name}, scaling={scaling_label}, elements/dpu={elems_per_dpu}, total={nr_elements}, dpus={nr_dpus} ---")
                    
                    # 4. Run Benchmarks
                    for bench in benchmarks:
                        # Call prepare with common args + op_param
                        bench.prepare(nr_dpus, nr_elements, op_param, warmup, iterations, check=check, load_ref=check)
                        
                        if bench.compile(verbose):
                            try:
                                out = bench.run(verbose, dpus=nr_dpus)
                            except TypeError:
                                out = bench.run(verbose)
                                
                            time_val = bench.parse_time(out)
                            if time_val is not None:
                                results[bench.name] = time_val
                                if check:
                                    res = bench.parse_verification(out)
                                    print(f"[{bench.name}] VERIFICATION {'SUCCESSFUL' if res is True else 'FAILED' if res is False else 'NOT PERFORMED'}")
                    
                    # 5. Write Results
                    with open(output_csv, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        for bench_name, time_val in results.items():
                            row = [op_name, elems_per_dpu, nr_elements, nr_dpus, bench_name, time_val, scaling_label, 
                                   1 if getattr(args, 'pipeline', False) else 0, 
                                   1 if getattr(args, 'logging', False) else 0]
                            if extra_cols:
                                row.extend(extra_cols.values())
                            writer.writerow(row)
                            
                    print("Results: " + ", ".join([f"{k}={v}ms" for k,v in results.items()]))
