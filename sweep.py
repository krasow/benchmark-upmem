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
ELEMENTS_LIST = [8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024]
DPUS_LIST = [64, 128, 256, 512]
OPERATIONS_LIST = [
    ("add", "(a + b)"),
    ("dos", "-(a + b)"),
    ("complex", "abs(-((a + b) - a))") 
]

OUTPUT_CSV = "sweep_results.csv"

# Global verbose flag (set via args)
VERBOSE = False

def update_param_h(file_path, replacements):
    with open(file_path, 'r') as f:
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
         match = re.search(r"baseline \(ms\):\s*([0-9.]+)", stdout)
    else:
        match = re.search(rf"{label}\s*\(ms\):\s*([0-9.]+)", stdout)
        
    if match:
        return float(match.group(1))
    return None

def generate_plot(csv_path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
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
    
    ops = df['operation'].unique()
    
    # Define consistent styles
    # Benchmarks: Colors and Line Styles
    # SimplePIM: Red, solid
    # LibVectorDPU: Blue, dashed
    # Baseline: Purple, dash-dot
    benchmark_styles = {
        'simplepim': {'color': '#E24A33', 'linestyle': '-', 'label_name': 'SimplePIM'},
        'libvectordpu': {'color': '#348ABD', 'linestyle': '-', 'label_name': 'LibVectorDPU'},
        'baseline': {'color': '#00B050', 'linestyle': '-', 'label_name': 'Baseline'}
    }

    # Elements: Markers
    # Assign a unique marker to each element size found in the data
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
            
            # Plot simplepim
            if not subset['simplepim_time_ms'].isna().all():
                style = benchmark_styles['simplepim']
                plt.plot(subset['dpus'], subset['simplepim_time_ms'], 
                         marker=marker, color=style['color'], linestyle=style['linestyle'])
            
            # Plot libvectordpu
            if not subset['libvectordpu_time_ms'].isna().all():
                style = benchmark_styles['libvectordpu']
                plt.plot(subset['dpus'], subset['libvectordpu_time_ms'], 
                         marker=marker, color=style['color'], linestyle=style['linestyle'])
    
            # Plot baseline
            if not subset['baseline_time_ms'].isna().all():
                style = benchmark_styles['baseline']
                plt.plot(subset['dpus'], subset['baseline_time_ms'], 
                         marker=marker, color=style['color'], linestyle=style['linestyle'])
    
        plt.title(f'Benchmark Performance Comparison ({op})', fontsize=14)
        plt.xlabel('Number of DPUs', fontsize=12)
        plt.ylabel('Execution Time (ms)', fontsize=12)
        
        # Custom Legend
        legend_elements = []
        
        # Custom Legend
        legend_elements = []
        
        # Benchmarks Header (handlelength=0 so text starts left)
        legend_elements.append(Line2D([], [], color='none', label=r'$\bf{Benchmarks}$'))
        for name, style in benchmark_styles.items():
            legend_elements.append(Line2D([0], [0], color=style['color'], lw=2, linestyle=style['linestyle'], label=style['label_name']))
        
        # Spacer
        legend_elements.append(Line2D([], [], color='none', label=' '))

        # Element Sizes Header
        legend_elements.append(Line2D([], [], color='none', label=r'$\bf{Elements}$'))
        for elem in elements_list:
            marker = element_markers.get(elem, 'o')
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=f"{elem:,}", 
                                          markerfacecolor='gray', markeredgecolor='gray', markersize=8))

        # Place legend outside
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout() # Adjust layout to make room for legend
        
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        # Use logarithmic scale for X-axis since DPUs cover a wide range
        plt.xscale('log', basex=2)
        
        # Set ticks to be the actual DPU counts for clarity
        plt.xticks(dpus_list, labels=[str(d) for d in dpus_list], fontsize=11)
        plt.minorticks_off()
        
        plot_path = f"sweep_plot_{op}.png"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()

def generate_bar_charts(csv_path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: pandas, matplotlib or numpy not installed. Skipping bar chart generation.")
        return

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Cannot generate bar charts.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: CSV is empty. Skipping bar charts.")
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
    
    dpus_list = sorted(df['dpus'].unique())
    ops = df['operation'].unique()
    elements_list = sorted(df['elements'].unique())

    # Create a plot for each DPU count
    for dpu_count in dpus_list:
        dpu_subset = df[df['dpus'] == dpu_count]
        
        if dpu_subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Configuration for grouped bars
        n_elements = len(elements_list)
        n_ops = len(ops)
        n_benchmarks = 3 # SimplePIM, LibVectorDPU, Baseline
        
        # Width of a single bar
        bar_width = 0.08
        
        # Calculate positions
        # Major groups: Elements
        # Minor groups: Operations
        # Bars: Benchmarks
        
        # We'll use a linear scale and place ticks manually
        # Element groups spaced by 1.0
        indices = np.arange(n_elements)
        
        # For each element group, we have n_ops subgroups
        # Each subgroup has n_benchmarks bars
        # Total width of an element group needs to fit n_ops * n_benchmarks * bar_width
        # Plus some padding between subgroups and between groups.
        
        # Offsets for each operation subgroup relative to the element center
        # Center of element group is at index i (0, 1, 2...)
        
        # Let's say we want subgroups centered around the element tick.
        # Total width of one operation subgroup = 3 * bar_width
        # Total width of all operations = n_ops * 3 * bar_width + (n_ops - 1) * padding
        
        # To simplify, we iterate and calculate x-coords for each bar
        
        # Colors - one per benchmark
        # Colors - one per benchmark
        colors = ['#E24A33', '#348ABD', '#00B050'] # ggplot style red, blue, green
        
        for i, element in enumerate(elements_list):
            for j, op in enumerate(ops):
                # Get data for this specific (element, op, dpu) tuple
                row = dpu_subset[(dpu_subset['elements'] == element) & (dpu_subset['operation'] == op)]
                
                if row.empty:
                    continue
                
                # Values
                val_pim = row['simplepim_time_ms'].values[0] if not pd.isna(row['simplepim_time_ms'].values[0]) else 0
                val_vec = row['libvectordpu_time_ms'].values[0] if not pd.isna(row['libvectordpu_time_ms'].values[0]) else 0
                val_base = row['baseline_time_ms'].values[0] if not pd.isna(row['baseline_time_ms'].values[0]) else 0
                
                # Base position for this element
                x_base = i 
                
                # Offset for this operation subgroup
                # Center ops around 0
                # If we have 3 ops: indices -1, 0, 1. width approx 4*bar_width?
                # Let's space operations by 4 * bar_width
                
                op_spacing = 3.5 * bar_width
                total_op_width = (n_ops - 1) * op_spacing
                op_offset = (j * op_spacing) - (total_op_width / 2)
                
                # Center of this operation subgroup is at x_base + op_offset
                # Benchmarks are centered around that
                # Benchmark offsets: -width, 0, +width
                
                # Plot bars
                # SimplePIM
                ax.bar(x_base + op_offset - bar_width, val_pim, bar_width, color=colors[0], 
                       label='SimplePIM' if i == 0 and j == 0 else "")
                # LibVectorDPU
                ax.bar(x_base + op_offset, val_vec, bar_width, color=colors[1], 
                       label='LibVectorDPU' if i == 0 and j == 0 else "")
                # Baseline
                ax.bar(x_base + op_offset + bar_width, val_base, bar_width, color=colors[2], 
                       label='Baseline' if i == 0 and j == 0 else "")
                
                # Add text label for operation below the subgroup
                # Use axes transform to place relative to the x-axis, independent of data range
                ax.text(x_base + op_offset, -0.08, op, ha='center', va='top', 
                        fontsize=10, rotation=0, transform=ax.get_xaxis_transform(), fontweight='bold')

        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_xlabel('Number of Elements', fontsize=12)
        ax.set_title(f'Benchmark Comparison by Element Size & Operation ({dpu_count} DPUs)', fontsize=14)
        ax.set_xticks(indices)
        ax.set_xticklabels([f"{e:,}" for e in elements_list], fontsize=11)
        
        # Legend with slightly larger font
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.5)

        fig.tight_layout()
        
        # Adjust bottom margin to make space for operation labels
        plt.subplots_adjust(bottom=0.15)
        
        plot_path = f"bar_plot_{dpu_count}_dpus.png"
        plt.savefig(plot_path)
        print(f"Bar chart saved to {plot_path}")
        plt.close(fig)

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
            fieldnames = ["operation", "elements", "dpus", "simplepim_time_ms", "libvectordpu_time_ms", "baseline_time_ms"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush() # Ensure header is written if we crash early

            for op_name, op_val in OPERATIONS_LIST:
                for nr_elements in ELEMENTS_LIST:
                    for nr_dpus in DPUS_LIST:
                        print(f"\n--- Sweeping: op={op_name}, elements={nr_elements}, dpus={nr_dpus} ---")
                        
                        try:
                            # --- simplepim ---
                            if VERBOSE: print("Processing simplepim...")
                            simplepim_param_path = os.path.join(SIMPLEPIM_DIR, "Param.h")
                            update_param_h(simplepim_param_path, {
                                "dpu_number": nr_dpus,
                                "nr_elements": nr_elements,
                                "OPERATION": op_val
                            })
                            
                            time_pim = None
                            if run_command("make clean && make", cwd=SIMPLEPIM_DIR) is not None:
                                stdout_pim = run_command("./bin/host", cwd=SIMPLEPIM_DIR)
                                time_pim = parse_time(stdout_pim, "simplepim")
                            
                            # --- libvectordpu ---
                            if VERBOSE: print("Processing libvectordpu...")
                            libvectordpu_param_path = os.path.join(LIBVECTORDPU_DIR, "Param.h")
                            update_param_h(libvectordpu_param_path, {
                                "N": nr_elements,
                                "OPERATION": op_val
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
                                "nr_elements": nr_elements,
                                "OPERATION": op_val
                            })
                            
                            time_base = None
                            if run_command("make clean && make", cwd=BASELINE_DIR) is not None:
                                stdout_base = run_command("./bin/host_baseline", cwd=BASELINE_DIR)
                                time_base = parse_time(stdout_base, "baseline")
                            
                            print(f"Results: simplepim={time_pim}ms, libvectordpu={time_vec}ms, baseline={time_base}ms")
                            
                            # Incremental write
                            writer.writerow({
                                "operation": op_name,
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
                            print(f"Unexpected error during sweep configuration ({op_name}, {nr_elements}, {nr_dpus}): {e}")
                            # Write what we have (even if Nones)
                            writer.writerow({
                                "operation": op_name,
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
        generate_bar_charts(OUTPUT_CSV)

if __name__ == "__main__":
    main()
