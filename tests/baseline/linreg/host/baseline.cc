#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#ifndef DPURT
#define DPURT
#include <dpu>
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include "timer.h"
#include "../Param.h"

typedef struct {
    uint32_t data_offset;
    uint32_t weights_offset;
    uint32_t results_offset;
    uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

void load_bin(const char* filename, void* data, size_t size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for reading\n", filename);
        exit(1);
    }
    fread(data, 1, size, f);
    fclose(f);
}

int main() {
    int nr_of_dpus = dpu_number;
    dpu_set_t dpu_set;

    CHECK_UPMEM(dpu_alloc(nr_of_dpus, "backend=hw", &dpu_set));
    CHECK_UPMEM(dpu_load(dpu_set, "./bin/baseline.dpu", nullptr));

    uint64_t elements_per_dpu = nr_elements / nr_of_dpus;
    uint32_t padded_row_size = ((dim + 1) * sizeof(T) + 7) & ~7;
    uint32_t weights_size = dim * sizeof(T);
    uint32_t red_size = dim * sizeof(RED_T);
    const int NR_TASKLETS = 12;

    DPU_LAUNCH_ARGS args[nr_of_dpus];
    for (int i = 0; i < nr_of_dpus; i++) {
        args[i].num_elements = elements_per_dpu;
        args[i].data_offset = 0;
        args[i].weights_offset = (elements_per_dpu * padded_row_size + 7) & ~7;
        args[i].results_offset = (args[i].weights_offset + weights_size + 7) & ~7;
    }

    T *all_elements = (T*)malloc(nr_elements * (padded_row_size / sizeof(T)) * sizeof(T));
    T *weights = (T*)malloc(dim * sizeof(T));
    int64_t *expected_grads = (int64_t*)malloc(dim * sizeof(int64_t));

    if (load_ref) {
        printf("Loading reference data from %s...\n", ref_path);
        T* col_buf = (T*)malloc(nr_elements * sizeof(T));
        char path[1024];
        for (int j = 0; j < dim; j++) {
            sprintf(path, "%s/ref_x_col_%d.bin", ref_path, j);
            load_bin(path, col_buf, nr_elements * sizeof(T));
            for (uint64_t i = 0; i < nr_elements; i++) {
                all_elements[i * (padded_row_size / sizeof(T)) + j] = col_buf[i];
            }
        }
        sprintf(path, "%s/ref_y.bin", ref_path);
        load_bin(path, col_buf, nr_elements * sizeof(T));
        for (uint64_t i = 0; i < nr_elements; i++) {
            all_elements[i * (padded_row_size / sizeof(T)) + dim] = col_buf[i];
        }
        sprintf(path, "%s/ref_grads.bin", ref_path);
        load_bin(path, expected_grads, dim * sizeof(int64_t));
        free(col_buf);
    } else {
        for (uint64_t i = 0; i < nr_elements; i++) {
            for (int j = 0; j < dim + 1; j++) {
                all_elements[i * (padded_row_size / sizeof(T)) + j] = (i + j) % 256;
            }
        }
    }

    for (int i = 0; i < dim; i++) weights[i] = 0;

    // Transfer inputs
    dpu_set_t dpu;
    uint32_t idx;
    DPU_FOREACH(dpu_set, dpu, idx) {
        CHECK_UPMEM(dpu_prepare_xfer(dpu, &all_elements[idx * elements_per_dpu * (padded_row_size / sizeof(T))]));
    }
    CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, args[0].data_offset, elements_per_dpu * padded_row_size, DPU_XFER_DEFAULT));

    // Transfer weights (broadcast)
    CHECK_UPMEM(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, args[0].weights_offset, weights, weights_size, DPU_XFER_DEFAULT));

    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
        DPU_FOREACH(dpu_set, dpu, idx) {
            CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx]));
        }
        CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0, sizeof(DPU_LAUNCH_ARGS), DPU_XFER_DEFAULT));
        CHECK_UPMEM(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    }

    Timer timer;
    start(&timer, 0, 0);

    RED_T *final_grads_accum = (RED_T*)malloc(dim * sizeof(RED_T));
    RED_T *dpu_tasklet_grads = (RED_T*)malloc(nr_of_dpus * NR_TASKLETS * dim * sizeof(RED_T));

    for (int it = 0; it < iterations; it++) {
        DPU_FOREACH(dpu_set, dpu, idx) {
            CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx]));
        }
        CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0, sizeof(DPU_LAUNCH_ARGS), DPU_XFER_DEFAULT));
        
        CHECK_UPMEM(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

        // Gather partial gradients from all tasklets on all DPUs
        DPU_FOREACH(dpu_set, dpu, idx) {
            CHECK_UPMEM(dpu_prepare_xfer(dpu, &dpu_tasklet_grads[idx * NR_TASKLETS * dim]));
        }
        CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, args[0].results_offset, NR_TASKLETS * red_size, DPU_XFER_DEFAULT));

        // CPU Reduction
        for (int j = 0; j < dim; j++) final_grads_accum[j] = 0;
        for (int d = 0; d < nr_of_dpus; d++) {
            for (int t = 0; t < NR_TASKLETS; t++) {
                for (int j = 0; j < dim; j++) {
                    final_grads_accum[j] += dpu_tasklet_grads[(d * NR_TASKLETS + t) * dim + j];
                }
            }
        }
    }

    stop(&timer, 0);

    printf("baseline (ms): ");
    print(&timer, 0, 1);
    printf("\n");

    if (check_correctness) {
        if (load_ref) {
            int match = 1;
            for (int j = 0; j < dim; j++) {
                if (final_grads_accum[j] != expected_grads[j]) {
                    printf("Mismatch at gradient %d: got %lld, expected %lld\n", j, (long long)final_grads_accum[j], (long long)expected_grads[j]);
                    match = 0;
                }
            }
            if (match) {
                printf("All results match after %d iterations.\n", iterations);
            }
        }
    }

    free(all_elements);
    free(weights);
    free(expected_grads);
    free(final_grads_accum);
    free(dpu_tasklet_grads);
    CHECK_UPMEM(dpu_free(dpu_set));

    return 0;
}
