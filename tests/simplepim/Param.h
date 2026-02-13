#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>
typedef int32_t T; 
const bool check_correctness = false;
const uint32_t dpu_number = 1024;
uint32_t print_info = 0;
uint64_t nr_elements = 2147483648;
int iterations = 50;
const int warmup_iterations = 10;
#define OPERATION(a, b) abs(-((a + b) - a))
#endif
