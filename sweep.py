import os
import re
import subprocess
import csv
import argparse
import sys

TESTS_DIR = "/scratch/david/benchmark-upmem/tests"
ENV_FILE = os.path.join(TESTS_DIR, ".localenv")

DEFAULT_ELEMENTS_PER_DPU = [2 * 1024 * 1024, 3 * 1024 * 1024]
# DEFAULT_STRONG_TOTAL_ELEMENTS = [128 * 1024 * 1024, 256 * 1024 * 1024]
DEFAULT_STRONG_TOTAL_ELEMENTS = [128 * 1024 * 1024]
DEFAULT_DPUS = [64, 128, 256, 512, 1024]
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

    def compile(self, verbose=False):
        """Runs make clean && make."""
        return self._run_shell("make clean && make", verbose)

    def run(self, verbose=False, env=None):
        """Runs the benchmark executable."""
        return self._run_shell(self.exec_cmd, verbose, env)

    def parse_time(self, stdout):
        """Parses execution time from stdout based on the label."""
        if not stdout:
            return None
        
        match = re.search(r"{0}\s*\(ms\):\s*([0-9.]+)".format(self.label), stdout)
            
        if match:
            return float(match.group(1))
        return None

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


class SimplePIM(Benchmark):
    def __init__(self):
        super().__init__("simplepim", "./bin/host")

    def prepare(self, dpus, elements, op_val, warmup):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup
        })

class LibVectorDPU(Benchmark):
    def __init__(self):
        super().__init__("libvectordpu", "./run")

    def prepare(self, dpus, elements, op_val, warmup):
        self.update_params_file({
            "N": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup
        })

    def run(self, verbose=False, dpus=None):
        # libVectordpu needs NR_DPUS env var for the run script
        env = {"NR_DPUS": str(dpus)} if dpus else {}
        return super(LibVectorDPU, self).run(verbose, env=env)

    def rebuild_library(self, use_pipeline, use_logging=False, verbose=False):
        """Rebuilds the libvectordpu library with the specified PIPELINE and LOGGING settings."""
        src_dir = os.path.join(os.path.dirname(TESTS_DIR), "opt", "vectordpu_src")
        dest_dir = os.path.join(os.path.dirname(TESTS_DIR), "opt", "vectordpu")
        
        pipeline_val = 1 if use_pipeline else 0
        logging_val = 1 if use_logging else 0
        command = "DESTDIR={0} make install BACKEND=hw PIPELINE={1} LOGGING={2} CXX_STANDARD=c++17".format(dest_dir, pipeline_val, logging_val)
        
        if verbose:
            print("[libvectordpu] Rebuilding library with PIPELINE={0}, LOGGING={1}...".format(pipeline_val, logging_val))
        
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

class Baseline(Benchmark):
    def __init__(self):
        super().__init__("baseline", "./bin/host_baseline")

    def prepare(self, dpus, elements, op_val, warmup):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup
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
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print("Error reading CSV: {0}".format(e))
            return

        # Check required columns
        required_cols = {"operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling"}
        # 'pipeline' and 'logging' are optional for backward compatibility
        if not required_cols.issubset(df.columns):
            print("CSV missing required columns. Expected: {0}".format(required_cols))
            return

        # Setup standard styles
        benchmark_colors = {
            "simplepim": "blue",
            "libvectordpu": "orange",
            "baseline": "green"
        }
        
        # Get unique elements_per_dpu for markers
        unique_elems_per_dpu = sorted(df['elements_per_dpu'].unique())
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        elem_markers = {elem: markers[i % len(markers)] for i, elem in enumerate(unique_elems_per_dpu)}

        operations = df['operation'].unique()

        for op in operations:
            op_data = df[df['operation'] == op]
            
            plt.figure(figsize=(10, 6))
            
            # Plot lines for each benchmark and elements_per_dpu/total_elements
            for bench in op_data['benchmark'].unique():
                bench_data = op_data[op_data['benchmark'] == bench]
                
                # Check if we have mixed scaling types in this op_data, or just one
                # Ideally we plot separate figures for weak vs strong, or handle them clearly.
                # For now, let's iterate over scaling types present.
                for scaling_type in bench_data['scaling'].unique():
                    scaling_data = bench_data[bench_data['scaling'] == scaling_type]
                    
                    if scaling_type == 'weak':
                        # Group by elements_per_dpu
                        for elem in scaling_data['elements_per_dpu'].unique():
                            subset = scaling_data[scaling_data['elements_per_dpu'] == elem]
                            subset = subset.sort_values(by='dpus')
                            
                            label = "{0} (Weak: {1:.1f}M/DPU)".format(bench, elem/1024/1024.0)
                            if 'pipeline' in subset.columns and subset['pipeline'].any():
                                label += " [PIPELINE]"
                            if 'logging' in subset.columns and subset['logging'].any():
                                label += " [LOGGING]"
                            
                            # Use elements_per_dpu for marker selection
                            marker = elem_markers.get(elem, 'o')
                            
                            plt.plot(subset['dpus'], subset['time'], 
                                     marker=marker,
                                     color=benchmark_colors.get(bench, 'gray'),
                                     label=label,
                                     linestyle='-' if bench == 'libvectordpu' else '--')
                    
                    elif scaling_type == 'strong':
                        # Group by total_elements
                        unique_total = sorted(scaling_data['total_elements'].unique())
                        # Re-map markers for total elements if needed, or just hash them
                        total_markers = {t: markers[i % len(markers)] for i, t in enumerate(unique_total)}
                        
                        for total in unique_total:
                            subset = scaling_data[scaling_data['total_elements'] == total]
                            subset = subset.sort_values(by='dpus')
                            
                            label = "{0} (Strong: {1:.0f}M Total)".format(bench, total/1024/1024.0)
                            if 'pipeline' in subset.columns and subset['pipeline'].any():
                                label += " [PIPELINE]"
                            if 'logging' in subset.columns and subset['logging'].any():
                                label += " [LOGGING]"
                                
                            plt.plot(subset['dpus'], subset['time'], 
                                     marker=total_markers.get(total, 'x'),
                                     color=benchmark_colors.get(bench, 'gray'),
                                     label=label,
                                     linestyle='-' if bench == 'libvectordpu' else '--')

            plt.title("Benchmark Performance ({0})".format(op))
            plt.xlabel("Number of DPUs")
            plt.ylabel("Execution Time (ms)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            plot_filename = "plot_{0}_{1}_scaling.png".format(op, scaling_type)
            plt.savefig(plot_filename)
            print("Plot saved to {0}".format(plot_filename))


class SweepRunner:
    def __init__(self, output_csv="sweep_results.csv", verbose=False, warmup=10, use_pipeline=False, use_logging=False, selected_benchmarks=None):
        self.output_csv = output_csv
        self.verbose = verbose
        self.warmup = warmup
        self.use_pipeline = use_pipeline
        self.use_logging = use_logging
        # Initialize Benchmarks
        self.simplepim = SimplePIM()
        self.libvectordpu = LibVectorDPU()
        self.baseline = Baseline()
        
        all_benchmarks = [
            ("simplepim", self.simplepim),
            ("libvectordpu", self.libvectordpu),
            ("baseline", self.baseline)
        ]
        
        if selected_benchmarks:
            self.benchmarks = [b for name, b in all_benchmarks if name in selected_benchmarks]
        else:
            self.benchmarks = [b for name, b in all_benchmarks]

    def run_sweep(self, operations=DEFAULT_OPERATIONS, elements_per_dpu_list=DEFAULT_ELEMENTS_PER_DPU, total_elements_list=DEFAULT_STRONG_TOTAL_ELEMENTS, dpus_list=DEFAULT_DPUS, scaling_mode="weak"):
        """
        Runs the parameter sweep.
        scaling_mode: 'weak' or 'strong'
        """
        # Rebuild libvectordpu library if needed and selected
        if any(isinstance(b, LibVectorDPU) for b in self.benchmarks):
            if not self.libvectordpu.rebuild_library(self.use_pipeline, self.use_logging, self.verbose):
                print("Failed to rebuild libvectordpu library. Aborting libvectordpu tests.")

        # Create CSV header if file doesn't exist
        file_exists = os.path.isfile(self.output_csv)
        with open(self.output_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling", "pipeline", "logging"])

        for op_name, op_val in operations:
            # Determine validation lists based on scaling mode
            # For weak scaling: iterate elements_per_dpu
            # For strong scaling: iterate total_elements
            
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

                    print("\n--- Sweeping: op={0}, scaling={1}, elements/dpu={2}, total={3}, dpus={4} ---".format(op_name, scaling_label, elems_per_dpu, nr_elements, nr_dpus))
                    
                    results = {}
                    
                    try:
                        # 1. SimplePIM
                        if self.simplepim in self.benchmarks:
                            if self.verbose: print("Processing simplepim...")
                            self.simplepim.prepare(nr_dpus, nr_elements, op_val, self.warmup)
                            time_pim = None
                            if self.simplepim.compile(self.verbose):
                                out = self.simplepim.run(self.verbose)
                                time_pim = self.simplepim.parse_time(out)
                            results["simplepim"] = time_pim

                        # 2. LibVectorDPU
                        if self.libvectordpu in self.benchmarks:
                            if self.verbose: print("Processing libvectordpu...")
                            self.libvectordpu.prepare(nr_dpus, nr_elements, op_val, self.warmup)
                            time_vec = None
                            if self.libvectordpu.compile(self.verbose):
                                out = self.libvectordpu.run(self.verbose, dpus=nr_dpus)
                                time_vec = self.libvectordpu.parse_time(out)
                            results["libvectordpu"] = time_vec

                        # 3. Baseline
                        if self.baseline in self.benchmarks:
                            if self.verbose: print("Processing baseline...")
                            self.baseline.prepare(nr_dpus, nr_elements, op_val, self.warmup)
                            time_base = None
                            if self.baseline.compile(self.verbose):
                                out = self.baseline.run(self.verbose)
                                time_base = self.baseline.parse_time(out)
                            results["baseline"] = time_base

                        # Log results to CSV
                        with open(self.output_csv, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            for bench_name, time_val in results.items():
                                if time_val is not None:
                                    is_pipeline = 1 if (bench_name == "libvectordpu" and self.use_pipeline) else 0
                                    is_logging = 1 if (bench_name == "libvectordpu" and self.use_logging) else 0
                                    writer.writerow([op_name, elems_per_dpu, nr_elements, nr_dpus, bench_name, time_val, scaling_label, is_pipeline, is_logging])
                        
                        results_str = ", ".join(["{0}={1}ms".format(k, v) for k, v in results.items() if v is not None])
                        print("Results: {0}".format(results_str))

                    except Exception as e:
                        print("Error during sweep step: {0}".format(e))
                        if self.verbose:
                            import traceback
                            traceback.print_exc()

        print("\nSweep completed. Results saved to {0}".format(self.output_csv))


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for UPMEM benchmarks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot from the results")
    parser.add_argument("--only-plot", action="store_true", help="Only generate the plot from existing sweep_results.csv")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--scaling", choices=["weak", "strong", "both"], default="weak", help="Scaling type: weak, strong, or both (default: weak)")
    parser.add_argument("--pipeline", action="store_true", help="Enable PIPELINE support for libvectordpu (requires rebuild)")
    parser.add_argument("--logging", action="store_true", help="Enable LOGGING support for libvectordpu (requires rebuild)")
    
    # Selective benchmarks
    parser.add_argument("--libvectordpu", action="store_true", help="Run only libvectordpu benchmark")
    parser.add_argument("--simplepim", action="store_true", help="Run only simplepim benchmark")
    parser.add_argument("--baseline", action="store_true", help="Run only baseline benchmark")
    
    args = parser.parse_args()

    # Determine CSV file path - use absolute path or current dir
    # Using simple filename as it matches original script behavior
    csv_file = "sweep_results.csv"

    runner = None
    if not args.only_plot:
        # Remove old CSV if starting a new run to avoid schema mismatch
        if os.path.exists(csv_file):
            print("Removing existing {0} to start fresh sweep.".format(csv_file))
            os.remove(csv_file)

        selected_benchmarks = []
        if args.libvectordpu: selected_benchmarks.append("libvectordpu")
        if args.simplepim: selected_benchmarks.append("simplepim")
        if args.baseline: selected_benchmarks.append("baseline")
        
        if not selected_benchmarks:
            selected_benchmarks = None # Default to all in constructor

        runner = SweepRunner(
            output_csv=csv_file, 
            verbose=args.verbose, 
            warmup=args.warmup, 
            use_pipeline=args.pipeline, 
            use_logging=args.logging,
            selected_benchmarks=selected_benchmarks
        )
        
        modes_to_run = []
        if args.scaling == "both":
            modes_to_run = ["weak", "strong"]
        else:
            modes_to_run = [args.scaling]

        for mode in modes_to_run:
            print("Starting {0} scaling sweep...".format(mode))
            runner.run_sweep(scaling_mode=mode)

    if args.plot or args.only_plot:
        print("Generating line plots...")
        plotter = Plotter(csv_file)
        plotter.plot()

if __name__ == "__main__":
    main()
