import os
import argparse
import sys
from benchmarks.core import SuiteRegistry, Plotter
from benchmarks import elementwise, linreg, pipeline_comp, interpreter_comp, kernel_loading, reduction_promotion, fusion_sweep

def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for UPMEM benchmarks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot from the results")
    parser.add_argument("--only-plot", action="store_true", help="Only generate the plot from existing sweep_results.csv")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--scaling", choices=["weak", "strong", "both"], default="weak", help="Scaling type: weak, strong, or both (default: weak)")
    parser.add_argument("--pipeline", action="store_true", help="Enable PIPELINE support for libvectordpu (requires rebuild)")
    parser.add_argument("--logging", action="store_true", help="Enable LOGGING support for libvectordpu (requires rebuild)")
    parser.add_argument("--jit", action="store_true", help="Enable JIT compilation for libvectordpu (requires rebuild)")
    parser.add_argument("--trace", nargs='?', const=True, default=False, help="Enable TRACE support (optionally specify output file)")
    parser.add_argument("--debug", action="store_true", help="Preserve JIT temporary files for debugging")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for the benchmark (default: 1)")
    parser.add_argument("--check", action="store_true", help="Verify correctness by comparing results")
    parser.add_argument("--bits", choices=["32", "64", "both"], default="32", help="Reduction bit-width: 32, 64, or both (default: 32)")
    parser.add_argument("--fusion-lookahead", type=int, nargs="+", default=[4], help="Max fusion lookahead length to sweep (default: 4)")
    parser.add_argument("--dpus", type=int, nargs="+", help="List of DPUs to sweep over")
    parser.add_argument("--skip-rebuild", dest="skip_rebuild", action="store_true", help="Skip rebuilding libraries")
    parser.add_argument("--csv-file", type=str, default="sweep_results.csv", help="CSV file to store results")
    parser.add_argument("--append", action="store_true", help="Append results to existing CSV instead of overwriting")
    # General Benchmark Filters
    parser.add_argument("--libvectordpu", action="store_true", help="Run only libvectordpu benchmark")
    parser.add_argument("--simplepim", action="store_true", help="Run only simplepim benchmark")
    parser.add_argument("--baseline", action="store_true", help="Run only baseline benchmark (if available)")
    
    # Initialize Registry
    registry = SuiteRegistry()
    elementwise.register(registry)
    linreg.register(registry)
    pipeline_comp.register(registry)
    interpreter_comp.register(registry)
    kernel_loading.register(registry)
    reduction_promotion.register(registry)
    fusion_sweep.register(registry)
    
    # Let suites add their own specific args if needed
    for suite_name in registry.list_suites():
        registry.get_suite(suite_name)["add_args"](parser)

    args = parser.parse_args()
    if args.logging or args.debug:
        args.verbose = True
    csv_file = args.csv_file

    if not args.only_plot:
        if os.path.exists(csv_file) and not args.append:
            print(f"Removing existing {csv_file} to start fresh sweep.")
            os.remove(csv_file)

        # Determine which suites to run
        suites_to_run = []
        if getattr(args, 'pipeline_comp', False):
            suites_to_run.append("pipeline_comp")
        if getattr(args, 'interpreter_comp', False):
            suites_to_run.append("interpreter_comp")
        if getattr(args, 'kernel_loading', False):
            suites_to_run.append("kernel_loading")
        if getattr(args, 'reduction_promotion', False):
            suites_to_run.append("reduction_promotion")
        if getattr(args, 'fusion_sweep', False):
            suites_to_run.append("fusion_sweep")
        
        if not suites_to_run:
            if getattr(args, 'linreg', False):
                suites_to_run.append("linreg")
            if getattr(args, 'elementwise', False) or not suites_to_run: # Default to elementwise
                if "elementwise" not in suites_to_run:
                    suites_to_run.append("elementwise")
        
        for suite_name in suites_to_run:
            print(f"--- Running Suite: {suite_name} ---")
            suite = registry.get_suite(suite_name)
            suite["runner"](args, suite)

    if args.plot or args.only_plot:
        if getattr(args, 'kernel_loading', False):
            kernel_loading.plot_results(csv_file)
        else:
            plotter = Plotter(csv_file)
            plotter.plot(bits_filter=getattr(args, 'bits', None))

if __name__ == "__main__":
    main()
