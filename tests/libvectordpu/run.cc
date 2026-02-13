#include <vectordpu.h>

#include <cstdlib>
#include <ctime>
#include <vector>
#include "Param.h"

inline dpu_vector<int> compute(const dpu_vector<int>& a, const dpu_vector<int>& b) {
    return OPERATION(a, b);
}   

void compare_cpu_dpu_vectors(const std::vector<int>& a,
                             const std::vector<int>& b,
                             const std::vector<int>& dpu_result,
                             uint32_t iterations) {
    const uint32_t N = a.size();
    std::vector<int> cpu_result(N);

    for (uint32_t i = 0; i < N; i++) {
        cpu_result[i] = OPERATION(a[i], b[i]);
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
    std::srand(std::time(nullptr));

    std::vector<T> a(N), b(N);
    for (uint32_t i = 0; i < N; i++) {
        a[i] = std::rand() % 10;  
        b[i] = std::rand() % 10; 	
    }

    // Timer timer2; 
    // start(&timer2, 0, 0);
    dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
    dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

    auto res = dpu_vector<T>(N); 
    res.add_fence();

    // stop(&timer2, 0);
    // printf("sending data first(ms): ");
    // print(&timer2, 0, 1);
    // printf("\n");

    for (uint32_t i = 0; i < warmup_iterations; i++) {
        res = (da + db);
    }

    Timer timer;
    start(&timer, 0, 0);

    for (uint32_t i = 0; i < iterations; i++) {
        res = (da + db);
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