#include <vectordpu.h>
#include <runtime.h>

#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>
#include "Param.h"

void load_bin(const std::string& filename, void* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << filename << " for reading" << std::endl;
        std::exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size);
}

int main() {
  {
    if (!load_ref) {
        std::srand(seed);
    }

    std::cout << "Starting libvectordpu_linreg with N=" << N << ", DIM=" << DIM << std::endl;

    std::vector<std::vector<T>> host_x_cols(DIM, std::vector<T>(N));
    std::vector<T> host_y(N);
    std::vector<int64_t> expected_grads(DIM);

    if (load_ref) {
        std::cout << "Loading reference data from " << ref_path << "..." << std::endl;
        for (uint32_t j = 0; j < DIM; j++) {
            load_bin(std::string(ref_path) + "/ref_x_col_" + std::to_string(j) + ".bin", host_x_cols[j].data(), N * sizeof(T));
        }
        load_bin(std::string(ref_path) + "/ref_y.bin", host_y.data(), N * sizeof(T));
        load_bin(std::string(ref_path) + "/ref_grads.bin", expected_grads.data(), DIM * sizeof(int64_t));
    } else {
        std::cout << "Generating host data..." << std::endl;
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < DIM; j++) {
                host_x_cols[j][i] = (i * (DIM + 1) + j) % 256;
            }
            host_y[i] = (i * (DIM + 1) + DIM) % 256;
        }
    }
    std::cout << "Host data generated. Transferring to DPU..." << std::endl;
    dpu_vector<T> dy = dpu_vector<T>::from_cpu(host_y);
    std::vector<dpu_vector<T>> dx_cols;
    for (uint32_t j = 0; j < DIM; j++) {
        dx_cols.push_back(dpu_vector<T>::from_cpu(host_x_cols[j]));
    }
    std::cout << "Transfer complete. Buffer sizes: " << N * sizeof(T) / 1024 / 1024 << " MiB per vector." << std::endl;

    // Weights initialized to 0
    std::cout << "Initializing DPU weights..." << std::endl;
    std::vector<T> dw_scalar(DIM, 0);

    std::cout << "Starting benchmarking loop (" << iterations << " iterations)..." << std::endl;
    Timer timer;
    start(&timer, 0, 0);

    std::vector<T> grads(DIM);
    for (uint32_t iter = 0; iter < iterations; iter++) {
        if (iterations > 1 && iter % 10 == 0) std::cout << "Iteration " << iter << "..." << std::endl;
        
        // 1. Compute Error = sum(X_j * w_j) - Y
        dpu_vector<T> error = -dy;
        for (uint32_t j = 0; j < DIM; j++) { 
            error = error + (dx_cols[j] * dw_scalar[j]);
        }
        
        // 2. Compute Gradients Grad_j = sum(X_j * error)
        for (uint32_t j = 0; j < DIM; j++) {
            grads[j] = sum((dx_cols[j] * error) >> (T)scaling_shift);
        }
    }

    stop(&timer, 0);

    std::cout << "Final gradients: ";
    for (uint32_t i = 0; i < DIM; i++) {
        std::cout << (int64_t)grads[i] << " ";
    }
    std::cout << std::endl;

    if (check_correctness) {
        bool match = true;
        if (load_ref) {
            for (uint32_t i = 0; i < DIM; i++) {
                if ((int64_t)grads[i] != expected_grads[i]) {
                    std::cout << "Mismatch at gradient " << i << ": got " << (int64_t)grads[i] 
                                << ", expected " << expected_grads[i] << std::endl;
                    match = false;
                }
            }
        }
        if (match) {
            std::cout << "All results match after " << iterations << " iterations." << std::endl;
        }
    }

    printf("libvectordpu (ms): ");
    print(&timer, 0, 1);
    printf("\n");
  }

    DpuRuntime::get().shutdown();
    return 0;
}
