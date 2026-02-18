import os
import csv
from .core import Benchmark, CPUBenchmark, LibVectorDPU
from .elementwise import DEFAULT_DPUS

class SimplePIMLinReg(Benchmark):
    def __init__(self):
        super().__init__("simplepim_linreg", "./bin/host", relative_dir="simplepim/linreg", label="simplepim")

    def prepare(self, dpus, elements, dim, iterations, warmup, check=False, load_ref=False, seed=1):
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

    def prepare(self, dpus, elements, dim, iterations, warmup, check=False, load_ref=False, seed=1):
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

def run_sweep(args, registry_config):
    output_csv = "sweep_results.csv"
    verbose = args.verbose
    warmup = args.warmup
    iterations = args.iterations
    dim = args.dim
    check = args.check
    
    simplepim = SimplePIMLinReg()
    libvectordpu = LibVectorDPULinReg()
    cpu_linreg = CPUBenchmark("linreg")
    
    selected_benchmarks = []
    if args.libvectordpu: selected_benchmarks.append(libvectordpu)
    if args.simplepim: selected_benchmarks.append(simplepim)
    if not selected_benchmarks:
        selected_benchmarks = [simplepim, libvectordpu]

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
    
    # Linreg specific defaults if not provided in args
    elements_per_dpu_list = [512*1024, 1024*1024]
    total_elements_list = [64*1024*1024]

    for mode in scaling_modes:
        sweep_configs = []
        if mode == "weak":
            for elems in elements_per_dpu_list:
                sweep_configs.append({"elems_per_dpu": elems, "total": None})
        else: # strong
            for total in total_elements_list:
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
                    print(f"\n--- Generating CPU Baseline for LinReg (dim={dim}, N={nr_elements}) ---")
                    libvectordpu.prepare(nr_dpus, nr_elements, dim, iterations, warmup, check=check, load_ref=True)
                    cpu_out = cpu_linreg.run(verbose)
                    cpu_time = cpu_linreg.parse_time(cpu_out)
                    if cpu_time: results["cpu_baseline"] = cpu_time

                print(f"\n--- LinReg Sweep: dim={dim}, scaling={scaling_label}, elements/dpu={elems_per_dpu}, total={nr_elements}, dpus={nr_dpus} ---")
                
                for bench in selected_benchmarks:
                    bench.prepare(nr_dpus, nr_elements, dim, iterations, warmup, check=check, load_ref=check)
                    if bench.compile(verbose):
                        out = bench.run(verbose, dpus=nr_dpus) if isinstance(bench, LibVectorDPULinReg) else bench.run(verbose)
                        time_val = bench.parse_time(out)
                        if time_val is not None:
                            results[bench.name] = time_val
                            if check:
                                res = bench.parse_verification(out)
                                print(f"[{bench.name}] VERIFICATION {'SUCCESSFUL' if res is True else 'FAILED' if res is False else 'NOT PERFORMED'}")

                with open(output_csv, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    for bench_name, time_val in results.items():
                        writer.writerow(["linreg", elems_per_dpu, nr_elements, nr_dpus, bench_name, time_val, scaling_label, 0, 0, dim, iterations])
                print("Results: " + ", ".join([f"{k}={v}ms" for k,v in results.items()]))

def register(registry):
    def add_args(parser):
        parser.add_argument("--linreg", action="store_true", help="Run linear regression benchmark")
        parser.add_argument("--dim", type=int, default=10, help="Dimension for linear regression (default: 10)")

    registry.register("linreg", run_sweep, add_args)
