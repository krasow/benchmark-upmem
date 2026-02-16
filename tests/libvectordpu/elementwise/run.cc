#include <vectordpu.h>

#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>
#include <iostream>
#include "Param.h"

inline dpu_vector<int> compute(const dpu_vector<int>& a, const dpu_vector<int>& b) {
    return OPERATION(a, b);
}   

void load_bin(const std::string& filename, void* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << filename << " for reading" << std::endl;
        std::exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size);
}

void compare_cpu_dpu_vectors(const std::vector<int>& a,
                             const std::vector<int>& b,
                             const std::vector<int>& dpu_result,
                             uint32_t iterations) {
    const uint32_t N = a.size();
    std::vector<int> cpu_result(N);

    if (load_ref) {
        std::cout << "Loading expected results from " << ref_path << "..." << std::endl;
        load_bin(std::string(ref_path) + "/ref_res.bin", cpu_result.data(), N * sizeof(T));
    } else {
        for (uint32_t i = 0; i < N; i++) {
            cpu_result[i] = OPERATION(a[i], b[i]);
        }
    }

    for (uint32_t i = 0; i < N; i++) {
        if (cpu_result[i] != dpu_result[i]) {
            std::cerr << "Mismatch at index " << i << ": CPU result = "
                      << cpu_result[i] << ", DPU result = " << dpu_result[i]
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    std::cout << "All results match after " << iterations
              << " iterations." << std::endl;
}

int main() {
    std::vector<T> a(N), b(N);
    if (load_ref) {
        std::cout << "Loading reference data from " << ref_path << "..." << std::endl;
        load_bin(std::string(ref_path) + "/ref_a.bin", a.data(), N * sizeof(T));
        load_bin(std::string(ref_path) + "/ref_b.bin", b.data(), N * sizeof(T));
    } else {
        std::srand(seed);
        for (uint32_t i = 0; i < N; i++) {
            a[i] = std::rand() % 10;
            b[i] = std::rand() % 10;
        }
    }

    dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
    dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

    auto res = dpu_vector<T>(N); 
    res.add_fence();

    for (uint32_t i = 0; i < warmup_iterations; i++) {
        res = compute(da, db);
    }

    Timer timer;
    start(&timer, 0, 0);

    for (uint32_t i = 0; i < iterations; i++) {
        res = compute(da, db);
    }

    std::vector<T> result = res.to_cpu();
    res.add_fence();

    stop(&timer, 0);

    printf("libvectordpu (ms): ");
    print(&timer, 0, 1);
    printf("\n");

    if (check_correctness){
        compare_cpu_dpu_vectors(a, b, result, iterations);
    }

    return 0;
}