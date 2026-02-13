import os
import re
import subprocess
import csv
import argparse
import sys

TESTS_DIR = "/scratch/david/benchmark-upmem/tests"
ENV_FILE = os.path.join(TESTS_DIR, ".localenv")

DEFAULT_ELEMENTS_PER_DPU = [2 * 1024 * 1024, 3 * 1024 * 1024]
DEFAULT_STRONG_TOTAL_ELEMENTS = [128 * 1024 * 1024, 256 * 1024 * 1024]
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
            print(f"Warning: Param file {self.param_file} not found.")
            return

        with open(self.param_file, 'r') as f:
            content = f.read()

        for pattern, value in replacements.items():
            if pattern == "OPERATION":
                # Handle macro replacement: #define OPERATION(a, b) ...
                regex = r"(#define\s+OPERATION\s*\(\w+,\s*\w+\)\s*)(.*)"
                content = re.sub(regex, rf"\1{value}", content)
            else:
                # Match assignments to const/uint32_t/uint64_t etc.
                regex = rf"(\b{pattern}\s*=\s*)([^;]+);"
                content = re.sub(regex, rf"\g<1>{value};", content)
        
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
        
        match = re.search(rf"{self.label}\s*\(ms\):\s*([0-9.]+)", stdout)
            
        if match:
            return float(match.group(1))
        return None

    def _run_shell(self, command, verbose=False, extra_env=None):
        """Executes a shell command with the necessary environment."""
        full_command = f"source {ENV_FILE} && {command}"
        if verbose:
            print(f"[{self.name}] Executing: {full_command}")
        
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
                    print(f"[{self.name}] STDOUT:\n{result.stdout}")
                if result.stderr:
                    print(f"[{self.name}] STDERR:\n{result.stderr}")

            if result.returncode != 0:
                print(f"Error running command: {command}")
                if not verbose:
                    print(result.stderr)
                return None
            return result.stdout
        except Exception as e:
            print(f"Subprocess exception: {e}")
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
        return super().run(verbose, env=env)

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
            print(f"CSV file {self.csv_path} not found.")
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
            print(f"Error reading CSV: {e}")
            return

        # Check required columns
        required_cols = {"operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling"}
        if not required_cols.issubset(df.columns):
            print(f"CSV missing required columns. Expected: {required_cols}")
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
                            
                            label = f"{bench} (Weak: {elem/1024/1024:.1f}M/DPU)"
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
                            
                            label = f"{bench} (Strong: {total/1024/1024:.0f}M Total)"
                            plt.plot(subset['dpus'], subset['time'], 
                                     marker=total_markers.get(total, 'x'),
                                     color=benchmark_colors.get(bench, 'gray'),
                                     label=label,
                                     linestyle='-' if bench == 'libvectordpu' else '--')

            plt.title(f"Benchmark Performance ({op})")
            plt.xlabel("Number of DPUs")
            plt.ylabel("Execution Time (ms)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            plot_filename = f"plot_{op}_{scaling_type}_scaling.png"
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")


class SweepRunner:
    def __init__(self, output_csv="sweep_results.csv", verbose=False, warmup=10):
        self.output_csv = output_csv
        self.verbose = verbose
        self.warmup = warmup
        # Initialize Benchmarks
        self.simplepim = SimplePIM()
        self.libvectordpu = LibVectorDPU()
        self.baseline = Baseline()
        self.benchmarks = [self.simplepim, self.libvectordpu, self.baseline]

    def run_sweep(self, operations=DEFAULT_OPERATIONS, elements_per_dpu_list=DEFAULT_ELEMENTS_PER_DPU, total_elements_list=DEFAULT_STRONG_TOTAL_ELEMENTS, dpus_list=DEFAULT_DPUS, scaling_mode="weak"):
        """
        Runs the parameter sweep.
        scaling_mode: 'weak' or 'strong'
        """
        # Create CSV header if file doesn't exist
        file_exists = os.path.isfile(self.output_csv)
        with open(self.output_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling"])

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

                    print(f"\n--- Sweeping: op={op_name}, scaling={scaling_label}, elements/dpu={elems_per_dpu}, total={nr_elements}, dpus={nr_dpus} ---")
                    
                    results = {}
                    
                    try:
                        # 1. SimplePIM
                        if self.verbose: print("Processing simplepim...")
                        self.simplepim.prepare(nr_dpus, nr_elements, op_val, self.warmup)
                        time_pim = None
                        if self.simplepim.compile(self.verbose):
                            out = self.simplepim.run(self.verbose)
                            time_pim = self.simplepim.parse_time(out)
                        results["simplepim"] = time_pim

                        # 2. LibVectorDPU
                        if self.verbose: print("Processing libvectordpu...")
                        self.libvectordpu.prepare(nr_dpus, nr_elements, op_val, self.warmup)
                        time_vec = None
                        if self.libvectordpu.compile(self.verbose):
                            out = self.libvectordpu.run(self.verbose, dpus=nr_dpus)
                            time_vec = self.libvectordpu.parse_time(out)
                        results["libvectordpu"] = time_vec

                        # 3. Baseline
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
                                    writer.writerow([op_name, elems_per_dpu, nr_elements, nr_dpus, bench_name, time_val, scaling_label])
                        
                        results_str = ", ".join([f"{k}={v}ms" for k, v in results.items() if v is not None])
                        print(f"Results: {results_str}")

                    except Exception as e:
                        print(f"Error during sweep step: {e}")
                        if self.verbose:
                            import traceback
                            traceback.print_exc()

        print(f"\nSweep completed. Results saved to {self.output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for UPMEM benchmarks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot from the results")
    parser.add_argument("--only-plot", action="store_true", help="Only generate the plot from existing sweep_results.csv")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--scaling", choices=["weak", "strong", "both"], default="weak", help="Scaling type: weak, strong, or both (default: weak)")
    args = parser.parse_args()

    # Determine CSV file path - use absolute path or current dir
    # Using simple filename as it matches original script behavior
    csv_file = "sweep_results.csv"

    runner = None
    if not args.only_plot:
        # Remove old CSV if starting a new run to avoid schema mismatch
        if os.path.exists(csv_file):
            print(f"Removing existing {csv_file} to start fresh sweep.")
            os.remove(csv_file)

        runner = SweepRunner(output_csv=csv_file, verbose=args.verbose, warmup=args.warmup)
        
        modes_to_run = []
        if args.scaling == "both":
            modes_to_run = ["weak", "strong"]
        else:
            modes_to_run = [args.scaling]

        for mode in modes_to_run:
            print(f"Starting {mode} scaling sweep...")
            runner.run_sweep(scaling_mode=mode)

    if args.plot or args.only_plot:
        print("Generating line plots...")
        plotter = Plotter(csv_file)
        plotter.plot()

if __name__ == "__main__":
    main()
