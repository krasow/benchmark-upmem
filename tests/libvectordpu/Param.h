#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>
typedef int32_t T; 
const bool check_correctness = false;
const uint64_t N = 3221225472;
const uint32_t iterations = 50;
const uint32_t warmup_iterations = 10;
#define OPERATION(a, b) abs(-((a + b) - a))
#endif