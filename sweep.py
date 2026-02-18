import os
import argparse
import sys
from benchmarks.core import SuiteRegistry, Plotter
from benchmarks import elementwise, linreg, pipeline_comp

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
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for the benchmark (default: 1)")
    parser.add_argument("--check", action="store_true", help="Verify correctness by comparing results")
    parser.add_argument("--dpus", type=int, nargs="+", help="List of DPUs to sweep over")
    
    # Initialize Registry
    registry = SuiteRegistry()
    elementwise.register(registry)
    linreg.register(registry)
    pipeline_comp.register(registry)
    
    # Let suites add their own specific args if needed
    for suite_name in registry.list_suites():
        registry.get_suite(suite_name)["add_args"](parser)

    args = parser.parse_args()
    csv_file = "sweep_results.csv"

    if not args.only_plot:
        if os.path.exists(csv_file):
            print(f"Removing existing {csv_file} to start fresh sweep.")
            os.remove(csv_file)

        # Determine which suites to run
        suites_to_run = []
        if args.pipeline_comp:
            suites_to_run.append("pipeline_comp")
        
        if not suites_to_run:
            if args.linreg:
                suites_to_run.append("linreg")
            if args.elementwise or not suites_to_run: # Default to elementwise
                if "elementwise" not in suites_to_run:
                    suites_to_run.append("elementwise")
        
        for suite_name in suites_to_run:
            print(f"--- Running Suite: {suite_name} ---")
            suite = registry.get_suite(suite_name)
            suite["runner"](args, suite)

    if args.plot or args.only_plot:
        plotter = Plotter(csv_file)
        plotter.plot()

if __name__ == "__main__":
    main()
