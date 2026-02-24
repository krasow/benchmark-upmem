import os
import csv
from .core import SuiteRegistry, execute_sweep, LibVectorDPU, DEFAULT_DPUS
from .linreg import SimplePIMLinReg, LibVectorDPULinReg, CPULinReg, BaselineLinReg

def run_sweep(args, registry_config):
    # Guard/Warning: This suite compares both 32 and 64 bits regardless of --bits flag
    if hasattr(args, 'bits') and args.bits != '32':
        print(f"\n[!WARNING] Suite 'reduction_promotion' ignores --bits={args.bits} and will compare BOTH 32-bit and 64-bit configurations.")

    output_csv = args.csv_file
    verbose = args.verbose
    warmup = args.warmup
    iterations = args.iterations
    dim = getattr(args, 'dim', 10)
    check = args.check
    
    simplepim = SimplePIMLinReg()
    libvectordpu = LibVectorDPULinReg()
    baseline = BaselineLinReg()
    cpu_linreg = CPULinReg("linreg")
    
    # Elements config
    elements_per_dpu_list = getattr(args, 'elements_per_dpu', [1024*1024])
    total_elements_list = getattr(args, 'total_elements', [64*1024*1024])

    # 1. Run libvectordpu with promotion=0
    print("\n>>> Phase 1: libvectordpu WITHOUT promotion (32-bit)")
    args.bits = "32"
    args.promotion = False
    libvectordpu.rebuild_library(args.pipeline, args.logging, args.trace, args.jit, getattr(args, 'debug', False), use_promotion=False, verbose=verbose)
    execute_sweep(args, [libvectordpu], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                  extra_cols={"dim": dim, "iterations": iterations},
                  elements_per_dpu_list=elements_per_dpu_list, total_elements_list=total_elements_list)
    
    # 2. Run libvectordpu with promotion=1
    print("\n>>> Phase 2: libvectordpu WITH promotion (64-bit)")
    args.bits = "64"
    args.promotion = True
    libvectordpu.rebuild_library(args.pipeline, args.logging, args.trace, args.jit, getattr(args, 'debug', False), use_promotion=True, verbose=verbose)
    execute_sweep(args, [libvectordpu], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                  extra_cols={"dim": dim, "iterations": iterations},
                  elements_per_dpu_list=elements_per_dpu_list, total_elements_list=total_elements_list)
    
    # 3. Run simplepim 32-bit
    print("\n>>> Phase 3: simplepim WITHOUT promotion (32-bit)")
    args.bits = "32"
    args.promotion = False 
    os.environ["EXTRA_DPU_FLAGS"] = "-DRED_T=int32_t"
    args.extra_flags = "-DRED_T=int32_t"
    execute_sweep(args, [simplepim], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                  extra_cols={"dim": dim, "iterations": iterations},
                  elements_per_dpu_list=elements_per_dpu_list, total_elements_list=total_elements_list)
    
    # 4. Run simplepim 64-bit
    print("\n>>> Phase 4: simplepim WITH promotion (64-bit)")
    args.bits = "64"
    args.promotion = True
    os.environ["EXTRA_DPU_FLAGS"] = "-DRED_T=int64_t"
    args.extra_flags = "-DRED_T=int64_t"
    execute_sweep(args, [simplepim], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                  extra_cols={"dim": dim, "iterations": iterations},
                  elements_per_dpu_list=elements_per_dpu_list, total_elements_list=total_elements_list)
    
    # 5. Run baseline 32-bit
    print("\n>>> Phase 5: baseline WITHOUT promotion (32-bit)")
    args.bits = "32"
    args.promotion = False
    os.environ["EXTRA_DPU_FLAGS"] = "-DRED_T=int32_t"
    args.extra_flags = "-DRED_T=int32_t"
    execute_sweep(args, [baseline], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                  extra_cols={"dim": dim, "iterations": iterations},
                  elements_per_dpu_list=elements_per_dpu_list, total_elements_list=total_elements_list)

    # 6. Run baseline 64-bit
    print("\n>>> Phase 6: baseline WITH promotion (64-bit)")
    args.bits = "64"
    args.promotion = True
    os.environ["EXTRA_DPU_FLAGS"] = "-DRED_T=int64_t"
    args.extra_flags = "-DRED_T=int64_t"
    execute_sweep(args, [baseline], [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                  extra_cols={"dim": dim, "iterations": iterations},
                  elements_per_dpu_list=elements_per_dpu_list, total_elements_list=total_elements_list)
    
    # Cleanup flags
    if "EXTRA_DPU_FLAGS" in os.environ:
        del os.environ["EXTRA_DPU_FLAGS"]
    if hasattr(args, "extra_flags"):
        del args.extra_flags

def register(registry):
    def add_args(parser):
        parser.add_argument("--reduction-promotion", action="store_true", help="Run reduction promotion comparison benchmark")

    registry.register("reduction_promotion", run_sweep, add_args)
