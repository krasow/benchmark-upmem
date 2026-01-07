#include <vectordpu.h>

#include <cstdlib>
#include <ctime>
#include <vector>

// Chain operations on DPU: ((a + b) - a) -> negate -> abs
inline dpu_vector<int> compute(const dpu_vector<int>& a, const dpu_vector<int>& b) {
    return abs(-((a + b) - a));
}   

void compare_cpu_dpu_vectors(const std::vector<int>& a,
                             const std::vector<int>& b,
                             const std::vector<int>& dpu_result,
                             uint32_t iterations) {
    const uint32_t N = a.size();
    std::vector<int> cpu_result(N);

    for (uint32_t i = 0; i < N; i++) {
        int temp = (a[i] + b[i]) - a[i];
        temp = -temp;
        cpu_result[i] = (temp < 0) ? -temp : temp;  
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

    const uint32_t N = 1024 * 1024;
    const uint32_t iterations = 100;

    std::vector<int> a(N), b(N);
    for (uint32_t i = 0; i < N; i++) {
        a[i] = std::rand() % 10;  
        b[i] = std::rand() % 10;  
    }
    dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
    dpu_vector<int> db = dpu_vector<int>::from_cpu(b);

    // undefined reference to `dpu_vector<int>::operator=(dpu_vector<int> const&)'
    auto res = dpu_vector<int>(N); 

    Timer timer;
    start(&timer, 0, 0);

    for (uint32_t i = 0; i < iterations; i++) {
        std::cout << "Iteration " << i + 1 << "/" << iterations << std::endl;
        res = compute(da, db);
    }

    std::vector<int> result = res.to_cpu();
    res.add_fence();

    stop(&timer, 0);

    printf("the total time with timing consumed is (ms): ");
    print(&timer, 0, 1);
    printf("\n");

    compare_cpu_dpu_vectors(a, b, result, iterations);

    return 0;
}