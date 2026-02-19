import os
import csv
from .core import Benchmark, TESTS_DIR, DEFAULT_DPUS

# The "complex" operation: abs(-((a + b) - a))
# baseline/elementwise uses this as a #define OPERATION macro (single fused kernel).
# baseline/elementwise-interpreter has it unrolled into separate loops per op,
# mimicking the interpreted kernel's step-by-step execution pattern.
COMPLEX_OP = ("complex", "abs(-((a + b) - a))")

DEFAULT_ELEMENTS_PER_DPU = [2 * 1024 * 1024, 3 * 1024 * 1024]
DEFAULT_STRONG_TOTAL_ELEMENTS = [128 * 1024 * 1024, 256 * 1024 * 1024]


class BaselineFused(Benchmark):
    """Compiled baseline with the complex op as a single fused OPERATION macro."""
    def __init__(self):
        super().__init__("baseline_fused", "./bin/host_baseline",
                         relative_dir="baseline/elementwise", label="baseline")

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "OPERATION": op_val,
            "warmup_iterations": warmup,
            "iterations": iterations,
            "check_correctness": "true" if check else "false",
            "load_ref": "true" if load_ref else "false",
            "seed": seed
        })


class BaselineInterpreter(Benchmark):
    """Compiled baseline with the complex op unrolled into separate loops (interpreter-style)."""
    def __init__(self):
        super().__init__("baseline_interpreter", "./bin/host_baseline",
                         relative_dir="baseline/elementwise-interpreter", label="baseline")

    def prepare(self, dpus, elements, op_val, warmup, iterations, check=False, load_ref=False, seed=1):
        self.update_params_file({
            "dpu_number": dpus,
            "nr_elements": elements,
            "warmup_iterations": warmup,
            "iterations": iterations,
        })


def run_sweep(args, registry_config):
    output_csv = args.csv_file
    verbose = args.verbose
    warmup = args.warmup
    iterations = args.iterations
    check = args.check

    fused = BaselineFused()
    interpreter = BaselineInterpreter()

    selected_benchmarks = [fused, interpreter]

    from .core import execute_sweep
    execute_sweep(args, selected_benchmarks, [COMPLEX_OP], metric_arg="op_val",
                  output_csv=output_csv,
                  extra_cols={"iterations": iterations},
                  elements_per_dpu_list=DEFAULT_ELEMENTS_PER_DPU,
                  total_elements_list=DEFAULT_STRONG_TOTAL_ELEMENTS)


def register(registry):
    def add_args(parser):
        parser.add_argument("--interpreter-comp", action="store_true",
                            help="Compare fused vs interpreter-style compiled kernels (complex op)")

    registry.register("interpreter_comp", run_sweep, add_args)
