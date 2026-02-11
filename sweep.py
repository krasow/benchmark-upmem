import os
import re
import subprocess
import csv
import argparse

# Configuration
TESTS_DIR = "/scratch/david/benchmark-upmem/tests"
SIMPLEPIM_DIR = os.path.join(TESTS_DIR, "simplepim")
LIBVECTORDPU_DIR = os.path.join(TESTS_DIR, "libvectordpu")
BASELINE_DIR = os.path.join(TESTS_DIR, "baseline")

# Default Sweep Parameters (can be adjusted)
ELEMENTS_LIST = [2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024]
DPUS_LIST = [8, 16, 32, 64, 128, 256, 512]

OUTPUT_CSV = "sweep_results.csv"

# Global verbose flag (set via args)
VERBOSE = False

def update_param_h(file_path, replacements):
    with open(file_path, 'r') as f:
        content = f.read()
    
    for pattern, value in replacements.items():
        # Match assignments to const/uint32_t/uint64_t etc.
        # Use \b to match whole word pattern.
        # Use \g<1> to avoid ambiguity with the replacement value digits.
        regex = rf"(\b{pattern}\s*=\s*)([^;]+);"
        content = re.sub(regex, rf"\g<1>{value};", content)
    
    with open(file_path, 'w') as f:
        f.write(content)

# Environment setup file
ENV_FILE = os.path.join(TESTS_DIR, ".localenv")

def run_command(command, cwd=None, env=None):
    # Prepend sourcing the environment file to the command
    full_command = f"source {ENV_FILE} && {command}"
    if VERBOSE:
        print(f"Executing: {full_command} in {cwd}")
    
    # Use bash explicitly to support 'source'
    try:
        result = subprocess.run(["/bin/bash", "-c", full_command], capture_output=True, text=True, cwd=cwd, env=env)
        
        if VERBOSE:
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)

        if result.returncode != 0:
            print(f"Error running command: {command}")
            if not VERBOSE:
                print(result.stderr)
            return None
        return result.stdout
    except Exception as e:
        print(f"Subprocess exception: {e}")
        return None

def parse_time(stdout, label):
    if not stdout:
        return None
    # Flexible match for the label followed by (ms): and then the number
    # Also handle the baseline output format "the total time with timing consumed is (ms): <time>"
    if label == "baseline":
         match = re.search(r"total time .* consumed is \(ms\):\s*([0-9.]+)", stdout)
    else:
        match = re.search(rf"{label}\s*\(ms\):\s*([0-9.]+)", stdout)
        
    if match:
        return float(match.group(1))
    return None

def generate_plot(csv_path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: pandas or matplotlib not installed. Skipping plot generation.")
        return

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Cannot generate plot.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: CSV is empty. Skipping plot.")
        return

    # Filter out rows where all times are None/NaN
    df = df.dropna(subset=['simplepim_time_ms', 'libvectordpu_time_ms', 'baseline_time_ms'], how='all')
    if df.empty:
        print("Error: No successful benchmark results to plot.")
        return

    # Use a style if available
    try:
        plt.style.use('ggplot')
    except:
        pass

    plt.figure(figsize=(12, 7))
    
    elements_list = sorted(df['elements'].unique())
    dpus_list = sorted(df['dpus'].unique())
    
    for elements in elements_list:
        subset = df[df['elements'] == elements].sort_values('dpus')
        
        # Plot simplepim
        if not subset['simplepim_time_ms'].isna().all():
            plt.plot(subset['dpus'], subset['simplepim_time_ms'], 
                     marker='o', linestyle='-', label=f'simplepim (elements={elements})')
        
        # Plot libvectordpu
        if not subset['libvectordpu_time_ms'].isna().all():
            plt.plot(subset['dpus'], subset['libvectordpu_time_ms'], 
                     marker='s', linestyle='--', label=f'libvectordpu (elements={elements})')

        # Plot baseline
        if not subset['baseline_time_ms'].isna().all():
            plt.plot(subset['dpus'], subset['baseline_time_ms'], 
                     marker='^', linestyle='-.', label=f'baseline (elements={elements})')

    plt.title('Benchmark Performance Comparison')
    plt.xlabel('Number of DPUs')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Use logarithmic scale for X-axis since DPUs cover a wide range
    plt.xscale('log', basex=2)
    
    # Set ticks to be the actual DPU counts for clarity
    plt.xticks(dpus_list, labels=[str(d) for d in dpus_list])
    plt.minorticks_off()
    
    plot_path = "sweep_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Parameter sweep for UPMEM benchmarks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot from the results")
    parser.add_argument("--only-plot", action="store_true", help="Only generate the plot from existing sweep_results.csv")
    args = parser.parse_args()
    VERBOSE = args.verbose

    if not args.only_plot:
        # Ensure UPMEM_HOME and SIMPLE_PIM_LIB are set if needed
        env = os.environ.copy()

        # Open CSV and write header immediately
        with open(OUTPUT_CSV, 'w', newline='') as f:
            fieldnames = ["elements", "dpus", "simplepim_time_ms", "libvectordpu_time_ms", "baseline_time_ms"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush() # Ensure header is written if we crash early

            for nr_elements in ELEMENTS_LIST:
                for nr_dpus in DPUS_LIST:
                    print(f"\n--- Sweeping: elements={nr_elements}, dpus={nr_dpus} ---")
                    
                    try:
                        # --- simplepim ---
                        if VERBOSE: print("Processing simplepim...")
                        simplepim_param_path = os.path.join(SIMPLEPIM_DIR, "Param.h")
                        update_param_h(simplepim_param_path, {
                            "dpu_number": nr_dpus,
                            "nr_elements": nr_elements
                        })
                        
                        time_pim = None
                        if run_command("make clean && make", cwd=SIMPLEPIM_DIR) is not None:
                            stdout_pim = run_command("./bin/host", cwd=SIMPLEPIM_DIR)
                            time_pim = parse_time(stdout_pim, "simplepim")
                        
                        # --- libvectordpu ---
                        if VERBOSE: print("Processing libvectordpu...")
                        libvectordpu_param_path = os.path.join(LIBVECTORDPU_DIR, "Param.h")
                        update_param_h(libvectordpu_param_path, {
                            "N": nr_elements
                        })
                        
                        time_vec = None
                        if run_command("make clean && make", cwd=LIBVECTORDPU_DIR) is not None:
                            # NR_DPUS is an env variable for libvectordpu
                            env["NR_DPUS"] = str(nr_dpus)
                            stdout_vec = run_command("./run", cwd=LIBVECTORDPU_DIR, env=env)
                            time_vec = parse_time(stdout_vec, "libvectordpu")
                            
                        # --- baseline ---
                        if VERBOSE: print("Processing baseline...")
                        baseline_param_path = os.path.join(BASELINE_DIR, "Param.h")
                        update_param_h(baseline_param_path, {
                            "dpu_number": nr_dpus,
                            "nr_elements": nr_elements
                        })
                        
                        time_base = None
                        if run_command("make clean && make", cwd=BASELINE_DIR) is not None:
                            stdout_base = run_command("./bin/host_baseline", cwd=BASELINE_DIR)
                            time_base = parse_time(stdout_base, "baseline")
                        
                        print(f"Results: simplepim={time_pim}ms, libvectordpu={time_vec}ms, baseline={time_base}ms")
                        
                        # Incremental write
                        writer.writerow({
                            "elements": nr_elements,
                            "dpus": nr_dpus,
                            "simplepim_time_ms": time_pim,
                            "libvectordpu_time_ms": time_vec,
                            "baseline_time_ms": time_base
                        })
                        f.flush() # Force write to disk
                    except KeyboardInterrupt:
                        print("\nSweep interrupted by user.")
                        return
                    except Exception as e:
                        print(f"Unexpected error during sweep configuration ({nr_elements}, {nr_dpus}): {e}")
                        # Write what we have (even if Nones)
                        writer.writerow({
                            "elements": nr_elements,
                            "dpus": nr_dpus,
                            "simplepim_time_ms": None,
                            "libvectordpu_time_ms": None,
                            "baseline_time_ms": None
                        })
                        f.flush()

        print(f"\nSweep completed. Results saved to {OUTPUT_CSV}")

    if args.plot or args.only_plot:
        generate_plot(OUTPUT_CSV)

if __name__ == "__main__":
    main()
