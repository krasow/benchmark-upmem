#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <dpu.h>
#include <omp.h>

#include "processing/map/Map.h"
#include "processing/zip/Zip.h"
#include "processing/ProcessingHelperHost.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "timer.h"
#include "Param.h"

void init(T* A, uint32_t salt){
    for (uint64_t i = 0; i < nr_elements; i++) {
        A[i] = (i + salt)%128;
    }
}

void vector_addition_host(T* A, T* B, T* res) {
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (uint64_t  i = 0; i < nr_elements; i++) {
        res[i] = abs(-((A[i] + B[i]) - A[i]));
    }
}

void run(){
    int iterations = 100;

    simplepim_management_t* table_management = table_management_init(dpu_number);
    T* A = (T*)malloc_scatter_aligned(nr_elements, sizeof(T), table_management);
    T* B = (T*)malloc_scatter_aligned(nr_elements, sizeof(T), table_management);

    T* correct_res = (T*)malloc((uint64_t)sizeof(T)*nr_elements);
    init(A, 0);
    init(B, 1);
    

    for (int i = 0; i <= iterations; i++) {
        vector_addition_host(A, B, correct_res);
    }

    Timer timer;
    start(&timer, 0, 0);
    simplepim_scatter("t1", A, nr_elements,  sizeof(T), table_management);
    simplepim_scatter("t2", B, nr_elements,  sizeof(T),  table_management);

    handle_t* add_handle = create_handle("daxby_funcs", MAP);
    handle_t* zip_handle = create_handle("", ZIP);

    table_zip("t1", "t2", "t3",  zip_handle, table_management);

    for (int i = 0; i <= iterations; i++) {
        table_map("t3", "t4", sizeof(T), add_handle, table_management, 0);
    }

    T* res = simplepim_gather("t4", table_management);

    stop(&timer, 0);

    printf("the total time with timing consumed is (ms): ");
    print(&timer, 0, 1);
    printf("\n");
    
    int32_t is_correct = 1;

    for(int i=0; i<nr_elements; i++){
        if(res[i]!=correct_res[i]){
            is_correct = 0;
            printf("result mismatch at position %d, got %d, expected %d \n", i, res[i], correct_res[i]);
            break;
        }
    } 

    if(is_correct){
        printf("the result is correct \n");
    }  
}

int main(int argc, char *argv[]){
  run();
  return 0;
}
