#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>

typedef int32_t T;
const uint64_t N = 67108864;
#define OPERATION(a, b) abs(-((a + b) - a))
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 10;
const uint32_t check_correctness = 1;
const uint32_t load_ref = 1;
const char* ref_path = "./data";
const uint32_t seed = 1;

#endif
