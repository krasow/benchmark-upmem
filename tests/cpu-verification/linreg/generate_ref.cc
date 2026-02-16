#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include "Param.h"

void write_bin(const std::string& filename, const void* data, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        std::exit(1);
    }
    out.write(reinterpret_cast<const char*>(data), size);
}

int main(int argc, char** argv) {
    std::srand(seed); // Use seed from Param.h
    
    std::cout << "Generating LinReg reference data for N=" << N << ", DIM=" << DIM << " using seed=" << seed << "..." << std::endl;
    
    std::vector<std::vector<T>> host_x_cols(DIM, std::vector<T>(N));
    std::vector<T> host_y(N);
    
    // 1. Generate Input Data
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < DIM; j++) {
            host_x_cols[j][i] = (i * (DIM + 1) + j) % 256;
        }
        host_y[i] = (i * (DIM + 1) + DIM) % 256;
    }

    // Benchmark CPU performance with OpenMP
    std::vector<int64_t> expected_grads(DIM, 0);
    std::vector<T> error(N);

    omp_set_num_threads(24);

    // Warmup
    for (uint32_t iter = 0; iter < warmup_iterations; iter++) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < N; i++) {
            error[i] = -host_y[i];
        }
        for (uint32_t j = 0; j < DIM; j++) {
            int64_t accum = 0;
            #pragma omp parallel for reduction(+:accum)
            for (uint32_t i = 0; i < N; i++) {
                int64_t prod = (int64_t)host_x_cols[j][i] * error[i];
                accum += (prod >> scaling_shift);
            }
            expected_grads[j] = accum;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t iter = 0; iter < iterations; iter++) {
        // Compute Error = -Y
        #pragma omp parallel for
        for (uint32_t i = 0; i < N; i++) {
            error[i] = -host_y[i];
        }
        
        // Compute Gradients
        for (uint32_t j = 0; j < DIM; j++) {
            int64_t accum = 0;
            #pragma omp parallel for reduction(+:accum)
            for (uint32_t i = 0; i < N; i++) {
                int64_t prod = (int64_t)host_x_cols[j][i] * error[i];
                accum += (prod >> scaling_shift);
            }
            expected_grads[j] = accum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = (end - start) / iterations;
    
    std::cout << "cpu_baseline (ms): " << elapsed.count() << std::endl;

    std::cout << "Writing binary files to ./data/ ..." << std::endl;
    for(uint32_t j = 0; j < DIM; j++) {
        std::string filename = "data/ref_x_col_" + std::to_string(j) + ".bin";
        write_bin(filename, host_x_cols[j].data(), N * sizeof(T));
    }
    write_bin("data/ref_y.bin", host_y.data(), N * sizeof(T));
    write_bin("data/ref_grads.bin", expected_grads.data(), DIM * sizeof(int64_t));
    
    std::cout << "LinReg Reference generation complete." << std::endl;
    return 0;
}
