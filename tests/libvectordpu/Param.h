#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>
typedef int32_t T; 
const bool check_correctness = false;
const uint32_t N = 67108864;
const uint32_t iterations = 1000;
#define OPERATION(a, b) abs(-((a + b) - a))
#endif