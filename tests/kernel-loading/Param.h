#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>
typedef int32_t T; 
bool large = true;
const uint32_t dpu_number = 1024;
int iterations = 1000;
#define OPERATION(a, b) abs(-((a + b) - a))
#endif
