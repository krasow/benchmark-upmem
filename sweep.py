import os
import re
import subprocess
import csv
import argparse
import sys

TESTS_DIR = "/scratch/david/benchmark-upmem/tests"
ENV_FILE = os.path.join(TESTS_DIR, ".localenv")

DEFAULT_ELEMENTS = [16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024]
DEFAULT_DPUS = [64, 128, 256, 512]
DEFAULT_OPERATIONS = [
    ("add", "(a + b)"),
    ("dos", "-(a + b)"),
    ("complex", "abs(-((a + b) - a))") 
]

class Benchmark:
    """Base class for a benchmark."""
    def __init__(self, name, relative_dir, exec_cmd, label, param_file="Param.h"):
        self.name = name
        self.dir = os.path.join(TESTS_DIR, relative_dir)
        self.exec_cmd = exec_cmd
        self.label = label
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
        super().__init__("simplepim", "simplepim", "./bin/host", "simplepim")

    def prepare(self, dpus, elements, op_val):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val
        })

class LibVectorDPU(Benchmark):
    def __init__(self):
        super().__init__("libvectordpu", "libvectordpu", "./run", "libvectordpu")

    def prepare(self, dpus, elements, op_val):
        self.update_params_file({
            "N": elements,
            "OPERATION": op_val
        })

    def run(self, verbose=False, dpus=None):
        # libVectordpu needs NR_DPUS env var for the run script
        env = {"NR_DPUS": str(dpus)} if dpus else {}
        return super().run(verbose, env=env)

class Baseline(Benchmark):
    def __init__(self):
        super().__init__("baseline", "baseline", "./bin/host_baseline", "baseline")

    def prepare(self, dpus, elements, op_val):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val
        })


class Plotter:
    """Handles parsing CSV logs and generating plots/charts."""
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def _load_data(self):
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            if df.empty:
                print("Error: CSV is empty.")
                return None
            # Filter valid rows
            df = df.dropna(subset=['simplepim_time_ms', 'libvectordpu_time_ms', 'baseline_time_ms'], how='all')
            if df.empty:
                print("Error: No successful benchmark results to plot.")
                return None
            return df
        except ImportError:
            print("Error: pandas not installed.")
            return None
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def generate_line_plots(self):
        df = self._load_data()
        if df is None: return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            plt.style.use('ggplot')
        except ImportError:
            return

        ops = df['operation'].unique()
        
        benchmark_styles = {
            'simplepim': {'color': '#E24A33', 'linestyle': '-', 'label_name': 'SimplePIM'},
            'libvectordpu': {'color': '#348ABD', 'linestyle': '-', 'label_name': 'LibVectorDPU'},
            'baseline': {'color': '#00B050', 'linestyle': '-', 'label_name': 'Baseline'}
        }

        all_elements = sorted(df['elements'].unique())
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
        element_markers = {elem: markers[i % len(markers)] for i, elem in enumerate(all_elements)}

        for op in ops:
            op_df = df[df['operation'] == op]
            plt.figure(figsize=(12, 7))
            
            elements_list = sorted(op_df['elements'].unique())
            dpus_list = sorted(op_df['dpus'].unique())
            
            for elements in elements_list:
                subset = op_df[op_df['elements'] == elements].sort_values('dpus')
                marker = element_markers.get(elements, 'o')
                
                for key, style in benchmark_styles.items():
                    col_name = f"{key}_time_ms"
                    if not subset[col_name].isna().all():
                        plt.plot(subset['dpus'], subset[col_name], 
                                 marker=marker, color=style['color'], linestyle=style['linestyle'])
        
            plt.title(f'Benchmark Performance Comparison ({op})', fontsize=14)
            plt.xlabel('Number of DPUs', fontsize=12)
            plt.ylabel('Execution Time (ms)', fontsize=12)
            
            # --- Custom Legend ---
            legend_elements = []
            
            # Benchmarks Header
            legend_elements.append(Line2D([], [], color='none', label=r'$\bf{Benchmarks}$'))
            for name, style in benchmark_styles.items():
                legend_elements.append(Line2D([0], [0], color=style['color'], lw=2, linestyle=style['linestyle'], label=style['label_name']))
            
            legend_elements.append(Line2D([], [], color='none', label=' '))

            # Element Sizes Header
            legend_elements.append(Line2D([], [], color='none', label=r'$\bf{Elements}$'))
            for elem in elements_list:
                marker = element_markers.get(elem, 'o')
                legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=f"{elem:,}", 
                                              markerfacecolor='gray', markeredgecolor='gray', markersize=8))

            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.xscale('log', basex=2)
            plt.xticks(dpus_list, labels=[str(d) for d in dpus_list], fontsize=11)
            plt.minorticks_off()
            
            plot_path = f"sweep_plot_{op}.png"
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            plt.close()

    def generate_bar_charts(self):
        df = self._load_data()
        if df is None: return

        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            plt.style.use('ggplot')
        except ImportError:
            return

        dpus_list = sorted(df['dpus'].unique())
        ops = df['operation'].unique()
        elements_list = sorted(df['elements'].unique())

        for dpu_count in dpus_list:
            dpu_subset = df[df['dpus'] == dpu_count]
            if dpu_subset.empty: continue

            fig, ax = plt.subplots(figsize=(14, 8))
            
            n_elements = len(elements_list)
            n_ops = len(ops)
            bar_width = 0.08
            indices = np.arange(n_elements)
            colors = ['#E24A33', '#348ABD', '#00B050'] 
            
            for i, element in enumerate(elements_list):
                for j, op in enumerate(ops):
                    row = dpu_subset[(dpu_subset['elements'] == element) & (dpu_subset['operation'] == op)]
                    if row.empty: continue
                    
                    val_pim = row['simplepim_time_ms'].values[0] if not pd.isna(row['simplepim_time_ms'].values[0]) else 0
                    val_vec = row['libvectordpu_time_ms'].values[0] if not pd.isna(row['libvectordpu_time_ms'].values[0]) else 0
                    val_base = row['baseline_time_ms'].values[0] if not pd.isna(row['baseline_time_ms'].values[0]) else 0
                    
                    x_base = i 
                    op_spacing = 3.5 * bar_width
                    total_op_width = (n_ops - 1) * op_spacing
                    op_offset = (j * op_spacing) - (total_op_width / 2)
                    
                    ax.bar(x_base + op_offset - bar_width, val_pim, bar_width, color=colors[0], 
                           label='SimplePIM' if i == 0 and j == 0 else "")
                    ax.bar(x_base + op_offset, val_vec, bar_width, color=colors[1], 
                           label='LibVectorDPU' if i == 0 and j == 0 else "")
                    ax.bar(x_base + op_offset + bar_width, val_base, bar_width, color=colors[2], 
                           label='Baseline' if i == 0 and j == 0 else "")
                    
                    ax.text(x_base + op_offset, -0.08, op, ha='center', va='top', 
                            fontsize=10, rotation=0, transform=ax.get_xaxis_transform(), fontweight='bold')

            ax.set_ylabel('Execution Time (ms)', fontsize=12)
            ax.set_xlabel('Number of Elements', fontsize=12)
            ax.set_title(f'Benchmark Comparison by Element Size & Operation ({dpu_count} DPUs)', fontsize=14)
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{e:,}" for e in elements_list], fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, axis='y', alpha=0.5)
            fig.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            
            plot_path = f"bar_plot_{dpu_count}_dpus.png"
            plt.savefig(plot_path)
            print(f"Bar chart saved to {plot_path}")
            plt.close(fig)


class SweepRunner:
    def __init__(self, output_csv="sweep_results.csv", verbose=False):
        self.output_csv = output_csv
        self.verbose = verbose
        # Initialize Benchmarks
        self.simplepim = SimplePIM()
        self.libvectordpu = LibVectorDPU()
        self.baseline = Baseline()
        self.benchmarks = [self.simplepim, self.libvectordpu, self.baseline]

    def run_sweep(self, operations=DEFAULT_OPERATIONS, elements_list=DEFAULT_ELEMENTS, dpus_list=DEFAULT_DPUS):
        """Main execution loop for the parameter sweep."""
        
        # Open CSV and write header
        with open(self.output_csv, 'w', newline='') as f:
            fieldnames = ["operation", "elements", "dpus", "simplepim_time_ms", "libvectordpu_time_ms", "baseline_time_ms"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()

            for op_name, op_val in operations:
                for nr_elements in elements_list:
                    for nr_dpus in dpus_list:
                        print(f"\n--- Sweeping: op={op_name}, elements={nr_elements}, dpus={nr_dpus} ---")
                        
                        try:
                            # 1. SimplePIM
                            if self.verbose: print("Processing simplepim...")
                            self.simplepim.prepare(nr_dpus, nr_elements, op_val)
                            time_pim = None
                            if self.simplepim.compile(self.verbose):
                                out = self.simplepim.run(self.verbose)
                                time_pim = self.simplepim.parse_time(out)

                            # 2. LibVectorDPU
                            if self.verbose: print("Processing libvectordpu...")
                            self.libvectordpu.prepare(nr_dpus, nr_elements, op_val)
                            time_vec = None
                            if self.libvectordpu.compile(self.verbose):
                                out = self.libvectordpu.run(self.verbose, dpus=nr_dpus)
                                time_vec = self.libvectordpu.parse_time(out)

                            # 3. Baseline
                            if self.verbose: print("Processing baseline...")
                            self.baseline.prepare(nr_dpus, nr_elements, op_val)
                            time_base = None
                            if self.baseline.compile(self.verbose):
                                out = self.baseline.run(self.verbose)
                                time_base = self.baseline.parse_time(out)

                            print(f"Results: simplepim={time_pim}ms, libvectordpu={time_vec}ms, baseline={time_base}ms")
                            
                            writer.writerow({
                                "operation": op_name,
                                "elements": nr_elements,
                                "dpus": nr_dpus,
                                "simplepim_time_ms": time_pim,
                                "libvectordpu_time_ms": time_vec,
                                "baseline_time_ms": time_base
                            })
                            f.flush()

                        except KeyboardInterrupt:
                            print("\nSweep interrupted by user.")
                            return
                        except Exception as e:
                            print(f"Unexpected error during sweep configuration ({op_name}, {nr_elements}, {nr_dpus}): {e}")
                            f.flush()

        print(f"\nSweep completed. Results saved to {self.output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for UPMEM benchmarks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot from the results")
    parser.add_argument("--only-plot", action="store_true", help="Only generate the plot from existing sweep_results.csv")
    args = parser.parse_args()

    # Determine CSV file path - use absolute path or current dir
    # Using simple filename as it matches original script behavior
    csv_file = "sweep_results.csv"

    runner = None
    if not args.only_plot:
        runner = SweepRunner(output_csv=csv_file, verbose=args.verbose)
        # Note: calling run_sweep without args uses default lists from top of file
        runner.run_sweep()

    if args.plot or args.only_plot:
        if os.path.exists(csv_file):
            plotter = Plotter(csv_file)
            print("Generating line plots...")
            plotter.generate_line_plots()
            print("Generating bar charts...")
            plotter.generate_bar_charts()
        else:
            print(f"File {csv_file} not found, cannot generate plots.")

if __name__ == "__main__":
    main()
