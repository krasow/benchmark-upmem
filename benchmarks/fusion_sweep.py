import os
import csv
from .core import execute_sweep, LibVectorDPU, DEFAULT_DPUS
from .linreg import LibVectorDPULinReg, CPULinReg
from .elementwise import LibVectorDPUElementwise, CPUElementwise, DEFAULT_OPERATIONS

def run_sweep(args, registry_config):
    output_csv = args.csv_file
    verbose = args.verbose
    warmup = args.warmup
    iterations = args.iterations
    check = args.check
    dim = getattr(args, 'dim', 10)
    
    # Force weak scaling for this sweep as requested
    original_scaling = args.scaling
    args.scaling = "weak"
    
    libvectordpu_linreg = LibVectorDPULinReg()
    libvectordpu_elementwise = LibVectorDPUElementwise()
    
    cpu_linreg = CPULinReg("linreg")
    cpu_elementwise = CPUElementwise("elementwise")
    
    # 1. Elements config - User requested "1 weak scaling data size"
    # We'll use 1MB per DPU as a reasonable default if not specified
    elements_per_dpu_linreg = [1024*512] # ~2MB per DPU for linreg (2 float arrays)
    elements_per_dpu_elem = [1024*1024]  # ~4MB per DPU for elementwise (a, b arrays)
    
    dpus_list = args.dpus if args.dpus else [64, 128, 256, 512, 1024]
    
    lookahead_list = getattr(args, 'fusion_lookahead', [1, 2, 4, 8])
    
    # We loop over bit-widths if 'both' is selected, otherwise just use args.bits
    bit_widths = ["32", "64"] if getattr(args, 'bits', '32') == "both" else [getattr(args, 'bits', '32')]

    for bw in bit_widths:
        use_promotion = (bw == "64")
        args.promotion = use_promotion
        
        # Set RED_T based on bit-width
        if not hasattr(args, 'extra_flags'):
            args.extra_flags = ""
        flags = args.extra_flags.split()
        flags = [f for f in flags if not f.startswith("-DRED_T=")]
        if use_promotion:
            flags.append("-DRED_T=int64_t")
        else:
            flags.append("-DRED_T=int32_t")
        args.extra_flags = " ".join(flags)

        for la in lookahead_list:
            args.current_fusion_lookahead = la
            
            # Sub-Phase A: Pipeline (no JIT)
            print(f"\n>>> Fusion Sweep: {bw}-bit, Lookahead={la}, Configuration=PIPELINE")
            args.pipeline = True
            args.jit = False
            args.bits = bw
            
            libvectordpu_linreg.rebuild_library(use_pipeline=True, use_jit=False, use_promotion=use_promotion, lookahead=la, verbose=verbose)
            
            # Elementwise Run
            print(f"Running Elementwise (Pipeline, LA={la})...")
            execute_sweep(args, [libvectordpu_elementwise], DEFAULT_OPERATIONS, metric_arg="op_val", output_csv=output_csv, cpu_benchmark=cpu_elementwise,
                          elements_per_dpu_list=elements_per_dpu_elem, total_elements_list=[])
            
            # Linreg Run
            print(f"Running LinReg (Pipeline, LA={la})...")
            execute_sweep(args, [libvectordpu_linreg], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                          extra_cols={"dim": dim, "iterations": iterations},
                          elements_per_dpu_list=elements_per_dpu_linreg, total_elements_list=[])

            # Sub-Phase B: JIT
            print(f"\n>>> Fusion Sweep: {bw}-bit, Lookahead={la}, Configuration=JIT")
            args.pipeline = True # JIT implies pipeline
            args.jit = True
            
            libvectordpu_linreg.rebuild_library(use_pipeline=True, use_jit=True, use_promotion=use_promotion, lookahead=la, verbose=verbose)
            
            # Elementwise Run
            print(f"Running Elementwise (JIT, LA={la})...")
            execute_sweep(args, [libvectordpu_elementwise], DEFAULT_OPERATIONS, metric_arg="op_val", output_csv=output_csv, cpu_benchmark=cpu_elementwise,
                          elements_per_dpu_list=elements_per_dpu_elem, total_elements_list=[])
            
            # Linreg Run
            print(f"Running LinReg (JIT, LA={la})...")
            execute_sweep(args, [libvectordpu_linreg], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                          extra_cols={"dim": dim, "iterations": iterations},
                          elements_per_dpu_list=elements_per_dpu_linreg, total_elements_list=[])

    # Restore original scaling
    args.scaling = original_scaling

def register(registry):
    def add_args(parser):
        parser.add_argument("--fusion-sweep", action="store_true", help="Run optimized fusion lookahead sweep (Pipeline then JIT)")

    registry.register("fusion_sweep", run_sweep, add_args)
