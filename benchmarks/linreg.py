import os
import csv
from .core import Benchmark, CPUBenchmark, LibVectorDPU, DEFAULT_DPUS

class SimplePIMLinReg(Benchmark):
    def __init__(self):
        super().__init__("simplepim_linreg", "./bin/host", relative_dir="simplepim/linreg", label="simplepim")

    def prepare(self, dpus, elements, dim, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "dim": dim,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "shift_amount": 0,
            "prevent_overflow_shift_amount": 12,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"../../cpu-verification/linreg/data"',
            "seed": seed
        })

class LibVectorDPULinReg(LibVectorDPU):
    def __init__(self):
        super().__init__(name="libvectordpu_linreg", exec_cmd="./run", relative_dir="libvectordpu/linreg", label="libvectordpu")

    def prepare(self, dpus, elements, dim, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "N": elements,
            "DIM": dim,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "scaling_shift": 12,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"../../cpu-verification/linreg/data"',
            "seed": seed
        })

class CPULinReg(CPUBenchmark):
    def prepare(self, dpus, elements, dim, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "N": elements,
            "DIM": dim,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "scaling_shift": 12,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"./data"',
            "seed": seed
        })

class BaselineLinReg(Benchmark):
    def __init__(self):
        super().__init__("baseline_linreg", "./bin/host_baseline", relative_dir="baseline/linreg", label="baseline")

    def prepare(self, dpus, elements, dim, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "dim": dim,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"../../cpu-verification/linreg/data"',
            "seed": seed
        })

def run_sweep(args, registry_config):
    output_csv = "sweep_results.csv"
    verbose = args.verbose
    warmup = args.warmup
    iterations = args.iterations
    dim = args.dim
    check = args.check
    
    simplepim = SimplePIMLinReg()
    libvectordpu = LibVectorDPULinReg()
    baseline = BaselineLinReg()
    cpu_linreg = CPULinReg("linreg")
    
    selected_benchmarks = []
    if args.libvectordpu: selected_benchmarks.append(libvectordpu)
    if args.simplepim: selected_benchmarks.append(simplepim)
    if getattr(args, 'baseline', False): selected_benchmarks.append(baseline)
    if not selected_benchmarks:
        selected_benchmarks = [simplepim, libvectordpu, baseline]

    skip_rebuild = getattr(args, 'skip_rebuild', False)
    bits_choice = getattr(args, 'bits', '32')
    
    if bits_choice == "both":
        bit_widths = ["32", "64"]
    else:
        bit_widths = [bits_choice]

    for bw in bit_widths:
        use_promotion = (bw == "64")
        args.promotion = use_promotion # Store for plotter variant creation

        la = getattr(args, 'current_fusion_lookahead', 4)
        if libvectordpu in selected_benchmarks and not skip_rebuild:
            if not libvectordpu.rebuild_library(args.pipeline, args.logging, args.trace, args.jit, getattr(args, 'debug', False), use_promotion=use_promotion, lookahead=la, verbose=verbose):
                print(f"Failed to rebuild libvectordpu library for {bw}-bit. Skipping libvectordpu tests.")
                selected_benchmarks.remove(libvectordpu)
        
        # Set RED_T based on bit-width for simplepim and baseline
        if not hasattr(args, 'extra_flags'):
            args.extra_flags = ""
        
        # Clear existing RED_T if any
        flags = args.extra_flags.split()
        flags = [f for f in flags if not f.startswith("-DRED_T=")]
        if use_promotion:
            flags.append("-DRED_T=int64_t")
        else:
            flags.append("-DRED_T=int32_t")
        args.extra_flags = " ".join(flags)

        # Linreg specific defaults if not provided in args
        elements_per_dpu_list = getattr(args, 'elements_per_dpu', [512*1024, 1024*1024])
        total_elements_list = getattr(args, 'total_elements', [64*1024*1024])
                
        from .core import execute_sweep
        execute_sweep(args, selected_benchmarks, [("linreg", dim)], metric_arg="dim", output_csv=output_csv, cpu_benchmark=cpu_linreg, 
                      extra_cols={"dim": dim, "iterations": iterations},
                      elements_per_dpu_list=elements_per_dpu_list, total_elements_list=total_elements_list)

def register(registry):
    def add_args(parser):
        parser.add_argument("--linreg", action="store_true", help="Run linear regression benchmark")
        parser.add_argument("--dim", type=int, default=10, help="Dimension for linear regression (default: 10)")

    registry.register("linreg", run_sweep, add_args)
