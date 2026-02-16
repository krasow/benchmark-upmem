#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <dpu.h>

#include "processing/gen_red/GenRed.h"
#include "processing/ProcessingHelperHost.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "timer.h"
#include "Param.h"

void init_data(T* elements, uint32_t num_elements, uint32_t d){
    for (size_t i = 0; i < num_elements * d; i++){
        elements[i] = (T)(i % 256);
    }
}

void load_bin(const char* filename, void* data, size_t size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for reading\n", filename);
        exit(1);
    }
    fread(data, 1, size, f);
    fclose(f);
}

int main(){
    if (!load_ref) {
        srand(1);
    }
    simplepim_management_t* table_management = table_management_init(dpu_number);
    if (print_info) printf("dim: %d, num_elem: %ld, iter: %d, lr: %f \n", dim, nr_elements, iterations, lr);

    // inputs
    T* elements = (T*)malloc_scatter_aligned(nr_elements, (dim+1)*sizeof(T), table_management);
    int64_t* expected_grads = (int64_t*)malloc(dim * sizeof(int64_t));

    if (load_ref) {
        printf("Loading reference data from %s...\n", ref_path);
        T* col_buf = (T*)malloc(nr_elements * sizeof(T));
        char path[1024];
        for (int j = 0; j < dim; j++) {
            sprintf(path, "%s/ref_x_col_%d.bin", ref_path, j);
            load_bin(path, col_buf, nr_elements * sizeof(T));
            // Interleave into elements array (matching simplepim layout)
            for (uint64_t i = 0; i < nr_elements; i++) {
                elements[i * (dim + 1) + j] = col_buf[i];
            }
        }
        sprintf(path, "%s/ref_y.bin", ref_path);
        load_bin(path, col_buf, nr_elements * sizeof(T));
        for (uint64_t i = 0; i < nr_elements; i++) {
            elements[i * (dim + 1) + dim] = col_buf[i];
        }
        sprintf(path, "%s/ref_grads.bin", ref_path);
        load_bin(path, expected_grads, dim * sizeof(int64_t));
        free(col_buf);
    } else {
        init_data(elements, nr_elements, dim+1);
    }
    
    // weights data
    T* weights = malloc_broadcast_aligned(1, sizeof(T)*dim, table_management);
    for(int i=0; i<dim; i++){
        weights[i] = 0;
    }

    simplepim_scatter("t1", elements, nr_elements, (dim+1)*sizeof(T), table_management);
    uint32_t data_offset = lookup_table("t1", table_management)->end; 
    simplepim_broadcast("t2", weights, 1, dim*sizeof(T),  table_management);
    uint32_t weights_offset = lookup_table("t2", table_management)->end; 

    handle_t* va_handle = create_handle("lin_reg_funcs", REDUCE);

    // Warmup
    for(int l=0; l<warmup_iterations; l++){
        int64_t* res = table_gen_red("t1", "t3",  dim*sizeof(int64_t), 1, va_handle, table_management, data_offset);
        free(res);
        simplepim_broadcast("t2", weights, 1, dim*sizeof(T), table_management);
    }

    Timer timer;
    start(&timer, 0, 0);

    int64_t* final_res = NULL;
    for(int l=0; l<iterations; l++){
        final_res = table_gen_red("t1", "t3",  dim*sizeof(int64_t), 1, va_handle, table_management, data_offset);
        if (l < iterations - 1) free(final_res);
    }

    stop(&timer, 0);

    if (final_res) {
        printf("Final gradients: ");
        for (int i = 0; i < dim; i++) {
            printf("%ld ", final_res[i]);
        }
        printf("\n");

        if (check_correctness && load_ref) {
            int match = 1;
            for (int i = 0; i < dim; i++) {
                if (final_res[i] != expected_grads[i]) {
                    printf("Mismatch at gradient %d: got %ld, expected %ld\n", i, final_res[i], expected_grads[i]);
                    match = 0;
                }
            }
            if (match) {
                printf("All results match after %d iterations.\n", iterations);
            }
        }
        free(final_res);
    }

    printf("simplepim (ms): ");
    print(&timer, 0, 1);
    printf("\n");

    return 0;
}
