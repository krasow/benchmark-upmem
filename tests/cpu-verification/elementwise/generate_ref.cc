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
    
    std::cout << "Generating reference data for N=" << N << " using seed=" << seed << "..." << std::endl;
    
    std::vector<T> a(N);
    std::vector<T> b(N);
    std::vector<T> res(N);
    
    // Data Init
    for (uint64_t i = 0; i < N; i++) {
        a[i] = std::rand() % 10;
        b[i] = std::rand() % 10;
    }

    // Benchmark CPU performance with OpenMP
    omp_set_num_threads(24);
    
    // Warmup
    for (uint32_t iter = 0; iter < warmup_iterations; iter++) {
        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i++) {
            res[i] = OPERATION(a[i], b[i]);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    for (uint32_t iter = 0; iter < iterations; iter++) {
        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i++) {
            res[i] = OPERATION(a[i], b[i]);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = (end - start) / iterations;
    
    std::cout << "cpu_baseline (ms): " << elapsed.count() << std::endl;
    
    std::cout << "Writing binary files to ./data/ ..." << std::endl;
    write_bin("data/ref_a.bin", a.data(), N * sizeof(T));
    write_bin("data/ref_b.bin", b.data(), N * sizeof(T));
    write_bin("data/ref_res.bin", res.data(), N * sizeof(T));
    
    std::cout << "Reference generation complete." << std::endl;
    return 0;
}
