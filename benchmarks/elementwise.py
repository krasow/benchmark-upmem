import os
import csv
from .core import Benchmark, CPUBenchmark, LibVectorDPU, TESTS_DIR, DEFAULT_DPUS

DEFAULT_STRONG_TOTAL_ELEMENTS = [128 * 1024 * 1024, 256 * 1024 * 1024]
DEFAULT_ELEMENTS_PER_DPU = [2 * 1024 * 1024, 3 * 1024 * 1024]

DEFAULT_OPERATIONS = [
    # ("add", "a + b"),
    # ("dos", "-(a + b)"),
    ("complex", "abs(-((a + b) - a))") 
]

class SimplePIM(Benchmark):
    def __init__(self):
        super().__init__("simplepim", "./bin/host", relative_dir="simplepim/elementwise")

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "true" if check else "false",
            "load_ref": "true" if load_ref else "false",
            "ref_path": '"../../cpu-verification/elementwise/data"',
            "seed": seed
        })

class LibVectorDPUElementwise(LibVectorDPU):
    def __init__(self):
        super().__init__(name="libvectordpu", exec_cmd="./run", relative_dir="libvectordpu/elementwise", label="libvectordpu")

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "N": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"../../cpu-verification/elementwise/data"',
            "seed": seed
        })

class Baseline(Benchmark):
    def __init__(self):
        super().__init__("baseline", "./bin/host_baseline", relative_dir="baseline/elementwise")

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "true" if check else "false",
            "load_ref": "true" if load_ref else "false",
            "ref_path": '"../../cpu-verification/elementwise/data"',
            "seed": seed
        })

class CPUElementwise(CPUBenchmark):
    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "N": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "1" if check else "0",
            "load_ref": "1" if load_ref else "0",
            "ref_path": '"./data"',
            "seed": seed
        })

def run_sweep(args, registry_config):
    output_csv = "sweep_results.csv"
    verbose = args.verbose
    warmup = args.warmup
    iterations = args.iterations
    check = args.check
    
    simplepim = SimplePIM()
    libvectordpu = LibVectorDPUElementwise()
    baseline = Baseline()
    cpu_elementwise = CPUElementwise("elementwise")
    
    selected_benchmarks = []
    if args.libvectordpu: selected_benchmarks.append(libvectordpu)
    if args.simplepim: selected_benchmarks.append(simplepim)
    if args.baseline: selected_benchmarks.append(baseline)
    if not selected_benchmarks:
        selected_benchmarks = [simplepim, libvectordpu, baseline]

    skip_rebuild = getattr(args, 'skip_rebuild', False)
    
    if libvectordpu in selected_benchmarks and not skip_rebuild:
        la = getattr(args, 'current_fusion_lookahead', 4)
        if not libvectordpu.rebuild_library(args.pipeline, args.logging, args.trace, args.jit, getattr(args, 'debug', False), lookahead=la, verbose=verbose):
            print("Failed to rebuild libvectordpu library. Aborting libvectordpu tests.")
            selected_benchmarks.remove(libvectordpu)
            
    from .core import execute_sweep
    execute_sweep(args, selected_benchmarks, DEFAULT_OPERATIONS, metric_arg="op_val", output_csv=output_csv, cpu_benchmark=cpu_elementwise,
                  elements_per_dpu_list=DEFAULT_ELEMENTS_PER_DPU, total_elements_list=DEFAULT_STRONG_TOTAL_ELEMENTS)

def register(registry):
    def add_args(parser):
        parser.add_argument("--elementwise", action="store_true", help="Run elementwise benchmarks (default)")
    
    registry.register("elementwise", run_sweep, add_args)
