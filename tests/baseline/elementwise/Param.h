#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>
typedef int32_t T; 
const uint32_t check_correctness = true;
const uint32_t load_ref = true;
const char* ref_path = "../../cpu-verification/elementwise/data";
const uint32_t seed = 1;
const uint32_t dpu_number = 128;
uint32_t print_info = 0;
uint64_t nr_elements = 268435456;
int iterations = 1;
const int warmup_iterations = 5;
#define OPERATION(a, b) abs(-((a + b) - a))
#endif
