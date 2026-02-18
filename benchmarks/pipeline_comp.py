import copy
from .core import LibVectorDPU

def run_sweep(args, registry_config, registry):
    verbose = args.verbose
    
    # We want to run elementwise and linreg (if requested or by default)
    # in both pipeline=0 and pipeline=1 modes.
    
    # Determine which sub-suites to run
    sub_suites = []
    if args.linreg: sub_suites.append("linreg")
    if args.elementwise or not sub_suites: sub_suites.append("elementwise")
    
    print(f"\n{'#'*60}")
    print(f"PIPELINE COMPARISON SUITE")
    print(f"{'#'*60}")
    print(f"Sub-suites to run: {sub_suites}")
    print(f"{'#'*60}\n")
    
    libvectordpu = LibVectorDPU()
    
    for use_pipeline in [False, True]:
        print(f"\n>>>> Starting Pipeline Comparison: PIPELINE={'ENABLED' if use_pipeline else 'DISABLED'} <<<<\n")
        
        # 1. Rebuild library
        if not libvectordpu.rebuild_library(use_pipeline, args.logging, args.trace, verbose):
            print(f"Failed to rebuild libvectordpu library with PIPELINE={use_pipeline}. skipping.")
            continue
            
        # 2. Run sub-suites
        import copy
        sub_args = copy.copy(args)
        sub_args.pipeline = use_pipeline
        sub_args.skip_rebuild = True
        
        for suite_name in sub_suites:
            print(f"--- Running {suite_name} under pipeline comparison ---")
            suite = registry.get_suite(suite_name)
            suite["runner"](sub_args, suite)

def register(registry):
    def add_args(parser):
        parser.add_argument("--pipeline-comp", action="store_true", help="Compare libvectordpu performance with and without pipeline")
    
    def runner_wrapper(args, suite_config):
        return run_sweep(args, suite_config, registry)

    registry.register("pipeline_comp", runner_wrapper, add_args)
