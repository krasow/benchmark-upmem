#ifndef PARAM_H
#define PARAM_H

#include <stdlib.h>
#include <stdint.h>

const uint32_t print_info = 0;
typedef int T; 

const uint32_t check_correctness = 0;
const uint32_t load_ref = 0;
const char* ref_path = "../../cpu-verification/linreg/data";
const uint32_t seed = 1;

const uint32_t dpu_number = 256;
const uint32_t dim = 5;
const uint64_t nr_elements = 134217728;
const uint32_t iterations = 50;
const uint32_t warmup_iterations = 10;
const float lr = 1e-4;
const uint32_t shift_amount = 0;
const uint32_t prevent_overflow_shift_amount = 12;

#endif