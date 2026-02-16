import os
import re
import subprocess
import csv
import argparse
import sys

TESTS_DIR = "/scratch/david/benchmark-upmem/tests"
ENV_FILE = os.path.join(TESTS_DIR, ".localenv")

DEFAULT_ELEMENTS_PER_DPU = [2 * 1024 * 1024, 3 * 1024 * 1024]
DEFAULT_STRONG_TOTAL_ELEMENTS = [128 * 1024 * 1024]
DEFAULT_DPUS = [64, 128, 256, 512, 1024]

# DEFAULT_ELEMENTS_PER_DPU = [1 * 1024 * 1024]
# DEFAULT_DPUS = [512]

DEFAULT_OPERATIONS = [
    ("add", "(a + b)"),
    ("dos", "-(a + b)"),
    ("complex", "abs(-((a + b) - a))") 
]

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

    def parse_gradients(self, stdout):
        """Parses gradients from stdout."""
        if not stdout:
            return None
        match = re.search(r"Final gradients:\s*(.*)", stdout)
        if match:
            try:
                return [int(x) for x in match.group(1).strip().split()]
            except:
                return None
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

    def prepare(self, nr_dpus, nr_elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        # Update Param.h in the suite directory
        # CPUBenchmark uses its own directory
        # For elementwise op_val is string, for linreg it's dummy
        pass

    def run(self, verbose=False):
        # We need to compile first
        if not self.compile(verbose):
            return None
        return super().run(verbose)

class SimplePIM(Benchmark):
    def __init__(self):
        super().__init__("simplepim", "./bin/host", relative_dir="simplepim/elementwise")

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "true" if check else "false",
            "load_ref": "true" if load_ref else "false",
            "ref_path": '"../../cpu-verification/elementwise/data"',
            "seed": seed
        })

class LibVectorDPU(Benchmark):
    def __init__(self, name="libvectordpu", exec_cmd="./run", relative_dir="libvectordpu/elementwise", label="libvectordpu"):
        super().__init__(name, exec_cmd, relative_dir, label)

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "N": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"../../cpu-verification/elementwise/data"',
            "seed": seed
        })

    def run(self, verbose=False, dpus=None):
        # libVectordpu needs NR_DPUS env var for the run script
        env = {"NR_DPUS": str(dpus)} if dpus else {}
        return super().run(verbose, env=env)

    def rebuild_library(self, use_pipeline, use_logging=False, use_trace=False, verbose=False):
        """Rebuilds the libvectordpu library with the specified PIPELINE, LOGGING and TRACE settings."""
        src_dir = os.path.join(os.path.dirname(TESTS_DIR), "opt", "vectordpu_src")
        dest_dir = os.path.join(os.path.dirname(TESTS_DIR), "opt", "vectordpu")
        
        pipeline_val = 1 if use_pipeline else 0
        logging_val = 1 if use_logging else 0
        trace_val = 1 if use_trace else 0
        command = "DESTDIR={0} make install BACKEND=hw PIPELINE={1} LOGGING={2} TRACE={3} CXX_STANDARD=c++17".format(dest_dir, pipeline_val, logging_val, trace_val)
        
        if verbose:
            print("[libvectordpu] Rebuilding library with PIPELINE={0}, LOGGING={1}, TRACE={2}...".format(pipeline_val, logging_val, trace_val))
        
        # We need to run this in the src_dir
        full_command = "source {0} && {1}".format(ENV_FILE, command)
        try:
            result = subprocess.run(
                ["/bin/bash", "-c", full_command], 
                capture_output=True, 
                text=True, 
                cwd=src_dir
            )
            if result.returncode != 0:
                print("Error rebuilding libvectordpu library:")
                print(result.stderr)
                return False
            if verbose:
                print(result.stdout)
            return True
        except Exception as e:
            print("Exception during library rebuild: {0}".format(e))
            return False

class SimplePIMLinReg(Benchmark):
    def __init__(self):
        super().__init__("simplepim_linreg", "./bin/host", relative_dir="simplepim/linreg", label="simplepim")

    def prepare(self, dpus, elements, dim, iterations, warmup, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "dim": dim,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "shift_amount": 0,
            "prevent_overflow_shift_amount": 12,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"../../cpu-verification/linreg/data"',
            "seed": seed
        })

class LibVectorDPULinReg(LibVectorDPU):
    def __init__(self):
        super().__init__(name="libvectordpu_linreg", exec_cmd="./run", relative_dir="libvectordpu/linreg", label="libvectordpu")

    def prepare(self, dpus, elements, dim, iterations, warmup, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "N": elements,
            "DIM": dim,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "scaling_shift": 12,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"../../cpu-verification/linreg/data"',
            "seed": seed
        })

class Baseline(Benchmark):
    def __init__(self):
        super().__init__("baseline", "./bin/host_baseline", relative_dir="baseline/elementwise")

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "true" if check else "false",
            "load_ref": "true" if load_ref else "false",
            "ref_path": '"../../cpu-verification/elementwise/data"',
            "seed": seed
        })

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

class SweepRunner:
    def __init__(self, output_csv="sweep_results.csv", verbose=False, warmup=10, iterations=1, use_pipeline=False, use_logging=False, use_trace=False, selected_benchmarks=None):
        self.output_csv = output_csv
        self.verbose = verbose
        self.warmup = warmup
        self.iterations = iterations
        self.use_pipeline = use_pipeline
        self.use_logging = use_logging
        self.use_trace = use_trace
        self.check = False
        # Initialize Benchmarks
        self.simplepim = SimplePIM()
        self.libvectordpu = LibVectorDPU()
        self.baseline = Baseline()
        self.simplepim_linreg = SimplePIMLinReg()
        self.libvectordpu_linreg = LibVectorDPULinReg()
        self.cpu_elementwise = CPUBenchmark("elementwise")
        self.cpu_linreg = CPUBenchmark("linreg")
        
        all_benchmarks = [
            ("simplepim", self.simplepim),
            ("libvectordpu", self.libvectordpu),
            ("baseline", self.baseline),
            ("simplepim_linreg", self.simplepim_linreg),
            ("libvectordpu_linreg", self.libvectordpu_linreg)
        ]
        
        if selected_benchmarks:
            self.benchmarks = [b for name, b in all_benchmarks if name in selected_benchmarks]
        else:
            self.benchmarks = [b for name, b in all_benchmarks]

    def run_sweep(self, operations=DEFAULT_OPERATIONS, elements_per_dpu_list=DEFAULT_ELEMENTS_PER_DPU, total_elements_list=DEFAULT_STRONG_TOTAL_ELEMENTS, dpus_list=DEFAULT_DPUS, scaling_mode="weak"):
        """Runs the parameter sweep."""
        if any(isinstance(b, LibVectorDPU) for b in self.benchmarks):
            if not self.libvectordpu.rebuild_library(self.use_pipeline, self.use_logging, self.use_trace, self.verbose):
                print("Failed to rebuild libvectordpu library. Aborting libvectordpu tests.")

        file_exists = os.path.isfile(self.output_csv)
        with open(self.output_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling", "pipeline", "logging", "dim", "iterations"])

        for op_name, op_val in operations:
            sweep_configs = []
            if scaling_mode == "weak":
                for elems in elements_per_dpu_list:
                    sweep_configs.append({"elems_per_dpu": elems, "total": None})
            elif scaling_mode == "strong":
                for total in total_elements_list:
                    sweep_configs.append({"elems_per_dpu": None, "total": total})
            
            for config in sweep_configs:
                for nr_dpus in dpus_list:
                    if scaling_mode == "weak":
                        elems_per_dpu = config["elems_per_dpu"]
                        nr_elements = nr_dpus * elems_per_dpu
                        scaling_label = "weak"
                    else: # strong
                        nr_elements = config["total"]
                        elems_per_dpu = nr_elements // nr_dpus
                        scaling_label = "strong"
                    
                    results = {}
                    if self.check:
                        # 1. Run CPU Baseline / Generate Truth
                        ref_key = (op_name, nr_elements)
                        if ref_key != getattr(self, "last_ref_key_elementwise", None):
                            print(f"\n--- Generating CPU Baseline & Shared Truth for {op_name} (N={nr_elements}) ---")
                            # For elementwise, we use the libvectordpu Param.h as source of truth for the generator
                            self.libvectordpu.prepare(nr_dpus, nr_elements, op_val, self.warmup, self.iterations, check=self.check, load_ref=True)
                            cpu_out = self.cpu_elementwise.run(self.verbose)
                            cpu_time = self.cpu_elementwise.parse_time(cpu_out)
                            if cpu_time:
                                print(f"[cpu_baseline] Time: {cpu_time:.3f}ms")
                                self.last_cpu_time_elementwise = cpu_time
                            self.last_ref_key_elementwise = ref_key
                        
                        if hasattr(self, "last_cpu_time_elementwise"):
                            results["cpu_baseline"] = self.last_cpu_time_elementwise

                    print("\n--- Sweeping: op={0}, scaling={1}, elements/dpu={2}, total={3}, dpus={4} ---".format(op_name, scaling_label, elems_per_dpu, nr_elements, nr_dpus))
                    try:
                        if self.simplepim in self.benchmarks:
                            self.simplepim.prepare(nr_dpus, nr_elements, op_val, self.warmup, self.iterations, check=self.check, load_ref=self.check)
                            if self.simplepim.compile(self.verbose):
                                out = self.simplepim.run(self.verbose)
                                results["simplepim"] = self.simplepim.parse_time(out)
                                simplepim_out = out

                        if self.libvectordpu in self.benchmarks:
                            self.libvectordpu.prepare(nr_dpus, nr_elements, op_val, self.warmup, self.iterations, check=self.check, load_ref=self.check)
                            if self.libvectordpu.compile(self.verbose):
                                out = self.libvectordpu.run(self.verbose, dpus=nr_dpus)
                                results["libvectordpu"] = self.libvectordpu.parse_time(out)
                                libvectordpu_out = out

                        if self.baseline in self.benchmarks:
                            # Baseline also needs load_ref if check is enabled
                            self.baseline.prepare(nr_dpus, nr_elements, op_val, self.warmup, self.iterations, check=self.check, load_ref=self.check)
                            if self.baseline.compile(self.verbose):
                                out = self.baseline.run(self.verbose)
                                results["baseline"] = self.baseline.parse_time(out)
                                baseline_out = out

                        if self.check:
                            if "simplepim" in results:
                                simplepim_check = self.simplepim.parse_verification(simplepim_out)
                                if simplepim_check is True:
                                    print("[simplepim] INTERNAL VERIFICATION SUCCESSFUL")
                                elif simplepim_check is False:
                                    print("[simplepim] INTERNAL VERIFICATION FAILED")

                            if "libvectordpu" in results:
                                libvectordpu_check = self.libvectordpu.parse_verification(libvectordpu_out)
                                if libvectordpu_check is True:
                                    print("[libvectordpu] INTERNAL VERIFICATION SUCCESSFUL")
                                elif libvectordpu_check is False:
                                    print("[libvectordpu] INTERNAL VERIFICATION FAILED")

                            if "baseline" in results:
                                baseline_check = self.baseline.parse_verification(baseline_out)
                                if baseline_check is True:
                                    print("[baseline] INTERNAL VERIFICATION SUCCESSFUL")
                                elif baseline_check is False:
                                    print("[baseline] INTERNAL VERIFICATION FAILED")

                        with open(self.output_csv, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            for bench_name, time_val in results.items():
                                if time_val is not None:
                                    is_p = 1 if (bench_name == "libvectordpu" and self.use_pipeline) else 0
                                    is_l = 1 if (bench_name == "libvectordpu" and self.use_logging) else 0
                                    writer.writerow([op_name, elems_per_dpu, nr_elements, nr_dpus, bench_name, time_val, scaling_label, is_p, is_l, 0, 0])
                        
                        print("Results: " + ", ".join(["{0}={1}ms".format(k,v) for k,v in results.items() if v is not None]))
                    except Exception as e:
                        print("Error: {0}".format(e))

    def run_linreg_sweep(self, dim_list, iterations=1, elements_per_dpu_list=[512*1024, 1024*1024], total_elements_list=[64*1024*1024], dpus_list=DEFAULT_DPUS, scaling_mode="weak"):
        """Runs the linear regression parameter sweep."""
        if any(isinstance(b, LibVectorDPU) for b in self.benchmarks):
            if not self.libvectordpu.rebuild_library(self.use_pipeline, self.use_logging, self.use_trace, self.verbose):
                print("Failed to rebuild libvectordpu library. Aborting libvectordpu tests.")

        file_exists = os.path.isfile(self.output_csv)
        with open(self.output_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling", "pipeline", "logging", "dim", "iterations"])

        for dim in dim_list:
            sweep_configs = []
            if scaling_mode == "weak":
                for elems in elements_per_dpu_list:
                    sweep_configs.append({"elems_per_dpu": elems, "total": None})
            elif scaling_mode == "strong":
                for total in total_elements_list:
                    sweep_configs.append({"elems_per_dpu": None, "total": total})
            
            for config in sweep_configs:
                for nr_dpus in dpus_list:
                    if scaling_mode == "weak":
                        elems_per_dpu = config["elems_per_dpu"]
                        nr_elements = nr_dpus * elems_per_dpu
                        scaling_label = "weak"
                    else: # strong
                        nr_elements = config["total"]
                        elems_per_dpu = nr_elements // nr_dpus
                        scaling_label = "strong"

                    results = {}
                    if self.check:
                        # 1. Run CPU Baseline / Generate Truth
                        ref_key = (dim, nr_elements)
                        if ref_key != getattr(self, "last_ref_key_linreg", None):
                            print(f"\n--- Generating CPU Baseline & Shared Truth for LinReg (dim={dim}, N={nr_elements}) ---")
                            self.libvectordpu_linreg.prepare(nr_dpus, nr_elements, dim, iterations, self.warmup, check=self.check, load_ref=True)
                            cpu_out = self.cpu_linreg.run(self.verbose)
                            cpu_time = self.cpu_linreg.parse_time(cpu_out)
                            if cpu_time:
                                print(f"[cpu_baseline] Time: {cpu_time:.3f}ms")
                                self.last_cpu_time_linreg = cpu_time
                            self.last_ref_key_linreg = ref_key
                        
                        if hasattr(self, "last_cpu_time_linreg"):
                            results["cpu_baseline"] = self.last_cpu_time_linreg

                    print("\n--- LinReg Sweep: dim={0}, scaling={1}, elements/dpu={2}, total={3}, dpus={4} ---".format(dim, scaling_label, elems_per_dpu, nr_elements, nr_dpus))
                    try:
                        if self.simplepim_linreg in self.benchmarks:
                            self.simplepim_linreg.prepare(nr_dpus, nr_elements, dim, iterations, self.warmup, check=self.check, load_ref=self.check)
                            if self.simplepim_linreg.compile(self.verbose):
                                out = self.simplepim_linreg.run(self.verbose)
                                results["simplepim_linreg"] = self.simplepim_linreg.parse_time(out)
                                pim_out = out

                        if self.libvectordpu_linreg in self.benchmarks:
                            self.libvectordpu_linreg.prepare(nr_dpus, nr_elements, dim, iterations, self.warmup, check=self.check, load_ref=self.check)
                            if self.libvectordpu_linreg.compile(self.verbose):
                                out = self.libvectordpu_linreg.run(self.verbose, dpus=nr_dpus)
                                results["libvectordpu_linreg"] = self.libvectordpu_linreg.parse_time(out)
                                vec_out = out

                        if self.check:
                            pim_res = self.simplepim_linreg.parse_verification(pim_out) if 'pim_out' in locals() else None
                            vec_res = self.libvectordpu_linreg.parse_verification(vec_out) if 'vec_out' in locals() else None
                            
                            if pim_res is True:
                                print(f"--- [simplepim] INTERNAL VERIFICATION SUCCESSFUL ---")
                            elif pim_res is False:
                                print(f"--- [simplepim] INTERNAL VERIFICATION FAILED ---")
                                
                            if vec_res is True:
                                print(f"--- [libvectordpu] INTERNAL VERIFICATION SUCCESSFUL ---")
                            elif vec_res is False:
                                print(f"--- [libvectordpu] INTERNAL VERIFICATION FAILED ---")

                        with open(self.output_csv, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            for bench_name, time_val in results.items():
                                if time_val is not None:
                                    writer.writerow(["linreg", elems_per_dpu, nr_elements, nr_dpus, bench_name, time_val, scaling_label, 0, 0, dim, iterations])
                        print("Results: " + ", ".join(["{0}={1}ms".format(k,v) for k,v in results.items() if v is not None]))
                    except Exception as e:
                        print("Error: {0}".format(e))

def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for UPMEM benchmarks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot from the results")
    parser.add_argument("--only-plot", action="store_true", help="Only generate the plot from existing sweep_results.csv")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--scaling", choices=["weak", "strong", "both"], default="weak", help="Scaling type: weak, strong, or both (default: weak)")
    parser.add_argument("--pipeline", action="store_true", help="Enable PIPELINE support for libvectordpu (requires rebuild)")
    parser.add_argument("--logging", action="store_true", help="Enable LOGGING support for libvectordpu (requires rebuild)")
    parser.add_argument("--trace", action="store_true", help="Enable TRACE support for libvectordpu (requires rebuild)")
    parser.add_argument("--elementwise", action="store_true", help="Run elementwise benchmarks (default)")
    parser.add_argument("--linreg", action="store_true", help="Run linear regression benchmark")
    parser.add_argument("--dim", type=int, default=10, help="Dimension for linear regression (default: 10)")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations for linear regression (default: 1)")
    parser.add_argument("--libvectordpu", action="store_true", help="Run only libvectordpu benchmark")
    parser.add_argument("--simplepim", action="store_true", help="Run only simplepim benchmark")
    parser.add_argument("--baseline", action="store_true", help="Run only baseline benchmark")
    parser.add_argument("--check", action="store_true", help="Verify correctness by comparing results")
    parser.add_argument("--dpus", type=int, nargs="+", help="List of DPUs to sweep over")
    
    args = parser.parse_args()
    csv_file = "sweep_results.csv"

    if not args.only_plot:
        if os.path.exists(csv_file):
            print("Removing existing {0} to start fresh sweep.".format(csv_file))
            os.remove(csv_file)

        selected_benchmarks = []
        if args.linreg:
            if args.libvectordpu or (not args.simplepim and not args.baseline):
                selected_benchmarks.append("libvectordpu_linreg")
            if args.simplepim or (not args.libvectordpu and not args.baseline):
                selected_benchmarks.append("simplepim_linreg")
        else:
            # Default or explicit elementwise
            if args.libvectordpu: selected_benchmarks.append("libvectordpu")
            if args.simplepim: selected_benchmarks.append("simplepim")
            if args.baseline: selected_benchmarks.append("baseline")
        
        runner = SweepRunner(
            output_csv=csv_file, 
            verbose=args.verbose, 
            warmup=args.warmup, 
            iterations=args.iterations,
            use_pipeline=args.pipeline, 
            use_logging=args.logging,
            use_trace=args.trace,
            selected_benchmarks=selected_benchmarks if selected_benchmarks else None
        )
        runner.check = args.check
        
        modes = ["weak", "strong"] if args.scaling == "both" else [args.scaling]
        dpus_list = args.dpus if args.dpus else DEFAULT_DPUS
        for mode in modes:
            if args.linreg:
                print("Starting {0} scaling Linear Regression sweep...".format(mode))
                runner.run_linreg_sweep(dim_list=[args.dim], iterations=args.iterations, scaling_mode=mode, dpus_list=dpus_list)
            else:
                print("Starting {0} scaling sweep...".format(mode))
                runner.run_sweep(scaling_mode=mode, dpus_list=dpus_list)

    if args.plot or args.only_plot:
        plotter = Plotter(csv_file)
        plotter.plot()

if __name__ == "__main__":
    main()
