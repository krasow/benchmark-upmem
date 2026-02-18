import os
import re
import subprocess
import csv
import sys

TESTS_DIR = "/scratch/david/benchmark-upmem/tests"
ENV_FILE = os.path.join(TESTS_DIR, ".localenv")

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

    def run(self, verbose=False, env=None):
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
        
        command = f"DESTDIR={dest_dir} make install BACKEND=hw PIPELINE={pipeline_val} LOGGING={logging_val} TRACE={trace_val} CXX_STANDARD=c++17"
        
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
        required_cols = {"operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling"}
        if not required_cols.issubset(df.columns):
            print("CSV missing required columns. Expected: {0}".format(required_cols))
            return

        # Setup standard styles
        benchmark_colors = {
            "simplepim": "#E24A33",   # Reddish
            "libvectordpu": "#348ABD", # Blueish
            "baseline": "#00B050",     # Greenish
            "simplepim_linreg": "#E24A33",
            "libvectordpu_linreg": "#348ABD"
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

                for bench in op_df['benchmark'].unique():
                    bench_df = op_df[op_df['benchmark'] == bench]
                    for group_val in unique_groups:
                        subset = bench_df[bench_df[group_col] == group_val].sort_values('dpus')
                        if not subset.empty:
                            if scaling_type == 'weak':
                                label = "{0} ({1}/dpu)".format(bench, group_val)
                            else:
                                label = "{0} (Total: {1})".format(bench, group_val)
                            
                            if 'pipeline' in subset.columns and 'logging' in subset.columns:
                                is_p = subset['pipeline'].iloc[0]
                                is_l = subset['logging'].iloc[0]
                                if is_p: label += " [PIPELINE]"
                                if is_l: label += " [LOGGING]"
                                
                            plt.plot(subset['dpus'], subset['time'], 
                                     marker=group_markers.get(group_val, 'x'),
                                     color=benchmark_colors.get(bench, 'gray'),
                                     label=label,
                                     linestyle='-' if 'vectordpu' in bench else '--')

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

            # 2. Advanced Clustered Bar Charts (One per DPU count)
            # Skip if only linreg results are present
            if all(op == 'linreg' for op in scaling_df['operation'].unique()):
                print(f"Skipping clustered bar charts for {scaling_type} scaling (linreg only).")
                continue

            try:
                # Use ggplot style if available
                try: 
                    plt.style.use('ggplot')
                except: 
                    pass

                pivot_df = scaling_df.pivot_table(index=['dpus', 'operation', group_col], 
                                                 columns='benchmark', 
                                                 values='time').reset_index()
                
                for b in ['simplepim', 'libvectordpu', 'baseline']:
                    if b not in pivot_df.columns: pivot_df[b] = np.nan

                dpus_list = sorted(pivot_df['dpus'].unique())
                ops = sorted(pivot_df['operation'].unique())
                elements_list = sorted(pivot_df[group_col].unique())

                for dpu_count in dpus_list:
                    dpu_subset = pivot_df[pivot_df['dpus'] == dpu_count]
                    if dpu_subset.empty: continue

                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    n_elements = len(elements_list)
                    n_ops = len(ops)
                    # Configuration for grouped bars
                    bar_width = 0.08
                    op_spacing = 3.5 * bar_width
                    total_op_width = (n_ops - 1) * op_spacing
                    
                    indices = np.arange(n_elements)
                    
                    for i, element in enumerate(elements_list):
                        for j, op in enumerate(ops):
                            row = dpu_subset[(dpu_subset[group_col] == element) & (dpu_subset['operation'] == op)]
                            if row.empty: continue
                            
                            val_pim = row['simplepim'].values[0] if not pd.isna(row['simplepim'].values[0]) else 0
                            val_vec = row['libvectordpu'].values[0] if not pd.isna(row['libvectordpu'].values[0]) else 0
                            val_base = row['baseline'].values[0] if not pd.isna(row['baseline'].values[0]) else 0
                            
                            x_base = i 
                            op_offset = (j * op_spacing) - (total_op_width / 2)
                            
                            # SimplePIM
                            ax.bar(x_base + op_offset - bar_width, val_pim, bar_width, color=benchmark_colors['simplepim'], 
                                   label='SimplePIM' if i == 0 and j == 0 else "")
                            # LibVectorDPU
                            ax.bar(x_base + op_offset, val_vec, bar_width, color=benchmark_colors['libvectordpu'], 
                                   label='LibVectorDPU' if i == 0 and j == 0 else "")
                            # Baseline
                            ax.bar(x_base + op_offset + bar_width, val_base, bar_width, color=benchmark_colors['baseline'], 
                                   label='Baseline' if i == 0 and j == 0 else "")
                            
                            # Add text label for operation
                            ax.text(x_base + op_offset, -0.06, op, ha='center', va='top', 
                                    fontsize=10, rotation=0, transform=ax.get_xaxis_transform(), fontweight='bold')

                    ax.set_ylabel('Execution Time (ms)', fontsize=12)
                    ax.set_xlabel('Elements/DPU' if scaling_type == 'weak' else 'Number of Elements', fontsize=12, labelpad=30)
                    ax.set_title(f'Benchmark Comparison by Element Size & Operation ({dpu_count} DPUs)', fontsize=14)
                    ax.set_xticks(indices)
                    ax.set_xticklabels([f"{e:,}" for e in elements_list], fontsize=11)
                    
                    ax.legend(fontsize=10)
                    ax.grid(True, axis='y', alpha=0.5)
                    fig.tight_layout()
                    plt.subplots_adjust(bottom=0.18)
                    
                    bar_filename = f"bar_plot_{scaling_type}_{dpu_count}_dpus.png"
                    plt.savefig(bar_filename)
                    print(f"Bar chart saved to {bar_filename}")
                    plt.close(fig)

            except Exception as e:
                print(f"Error generating advanced bar charts: {e}")
