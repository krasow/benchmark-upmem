import os
import csv
from .core import Benchmark, CPUBenchmark, LibVectorDPU, TESTS_DIR

DEFAULT_STRONG_TOTAL_ELEMENTS = [128 * 1024 * 1024, 256 * 1024 * 1024]
DEFAULT_ELEMENTS_PER_DPU = [1 * 1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024]
DEFAULT_DPUS = [64, 128, 256, 512, 1024, 2048]

DEFAULT_OPERATIONS = [
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

def run_sweep(args, registry_config):
    output_csv = "sweep_results.csv"
    verbose = args.verbose
    warmup = args.warmup
    iterations = args.iterations
    check = args.check
    
    simplepim = SimplePIM()
    libvectordpu = LibVectorDPUElementwise()
    baseline = Baseline()
    cpu_elementwise = CPUBenchmark("elementwise")
    
    selected_benchmarks = []
    if args.libvectordpu: selected_benchmarks.append(libvectordpu)
    if args.simplepim: selected_benchmarks.append(simplepim)
    if args.baseline: selected_benchmarks.append(baseline)
    if not selected_benchmarks:
        selected_benchmarks = [simplepim, libvectordpu, baseline]

    skip_rebuild = getattr(args, 'skip_rebuild', False)
    if libvectordpu in selected_benchmarks and not skip_rebuild:
        if not libvectordpu.rebuild_library(args.pipeline, args.logging, args.trace, verbose):
            print("Failed to rebuild libvectordpu library. Aborting libvectordpu tests.")
            selected_benchmarks.remove(libvectordpu)

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["operation", "elements_per_dpu", "total_elements", "dpus", "benchmark", "time", "scaling", "pipeline", "logging", "dim", "iterations"])

    dpus_list = args.dpus if args.dpus else DEFAULT_DPUS
    scaling_modes = ["weak", "strong"] if args.scaling == "both" else [args.scaling]
    
    for mode in scaling_modes:
        for op_name, op_val in DEFAULT_OPERATIONS:
            sweep_configs = []
            if mode == "weak":
                for elems in DEFAULT_ELEMENTS_PER_DPU:
                    sweep_configs.append({"elems_per_dpu": elems, "total": None})
            else: # strong
                for total in DEFAULT_STRONG_TOTAL_ELEMENTS:
                    sweep_configs.append({"elems_per_dpu": None, "total": total})
            
            for config in sweep_configs:
                for nr_dpus in dpus_list:
                    if mode == "weak":
                        elems_per_dpu = config["elems_per_dpu"]
                        nr_elements = nr_dpus * elems_per_dpu
                        scaling_label = "weak"
                    else:
                        nr_elements = config["total"]
                        elems_per_dpu = nr_elements // nr_dpus
                        scaling_label = "strong"
                    
                    results = {}
                    if check:
                        print(f"\n--- Generating CPU Baseline for {op_name} (N={nr_elements}) ---")
                        libvectordpu.prepare(nr_dpus, nr_elements, op_val, warmup, iterations, check=check, load_ref=True)
                        cpu_out = cpu_elementwise.run(verbose)
                        cpu_time = cpu_elementwise.parse_time(cpu_out)
                        if cpu_time: results["cpu_baseline"] = cpu_time

                    print(f"\n--- Sweeping: op={op_name}, scaling={scaling_label}, elements/dpu={elems_per_dpu}, total={nr_elements}, dpus={nr_dpus} ---")
                    
                    for bench in selected_benchmarks:
                        bench.prepare(nr_dpus, nr_elements, op_val, warmup, iterations, check=check, load_ref=check)
                        if bench.compile(verbose):
                            out = bench.run(verbose, dpus=nr_dpus) if isinstance(bench, LibVectorDPUElementwise) else bench.run(verbose)
                            time_val = bench.parse_time(out)
                            if time_val is not None:
                                results[bench.name] = time_val
                                if check:
                                    res = bench.parse_verification(out)
                                    print(f"[{bench.name}] VERIFICATION {'SUCCESSFUL' if res is True else 'FAILED' if res is False else 'NOT PERFORMED'}")

                    with open(output_csv, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        for bench_name, time_val in results.items():
                            is_p = 1 if (bench_name == "libvectordpu" and args.pipeline) else 0
                            is_l = 1 if (bench_name == "libvectordpu" and args.logging) else 0
                            writer.writerow([op_name, elems_per_dpu, nr_elements, nr_dpus, bench_name, time_val, scaling_label, is_p, is_l, 0, 0])
                    print("Results: " + ", ".join([f"{k}={v}ms" for k,v in results.items()]))

def register(registry):
    def add_args(parser):
        parser.add_argument("--elementwise", action="store_true", help="Run elementwise benchmarks (default)")
        parser.add_argument("--libvectordpu", action="store_true", help="Run only libvectordpu benchmark")
        parser.add_argument("--simplepim", action="store_true", help="Run only simplepim benchmark")
        parser.add_argument("--baseline", action="store_true", help="Run only baseline benchmark")
    
    registry.register("elementwise", run_sweep, add_args)
