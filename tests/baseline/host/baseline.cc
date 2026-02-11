#include <stdio.h>
#include <stdlib.h> 
#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include "timer.h"
#include "../Param.h"

typedef struct {
	uint32_t lhs_offset;
	uint32_t rhs_offset;
	uint32_t res_offset;
	uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

void vec_xfer_to_dpu(dpu_set_t dpu_set, char* cpu, DPU_LAUNCH_ARGS *args) {
	dpu_set_t dpu;

	uint32_t idx_dpu = 0;
	size_t element = 0;

	DPU_FOREACH(dpu_set, dpu, idx_dpu) {
		CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu[element])));
		element += args[idx_dpu].num_elements * sizeof(int32_t);
	}

	uint32_t mram_location = args[0].res_offset;
	size_t xfer_size = args[0].num_elements * sizeof(int32_t);
	CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
				DPU_MRAM_HEAP_POINTER_NAME, mram_location,
				xfer_size, DPU_XFER_DEFAULT));
}

void vec_xfer_from_dpu(dpu_set_t dpu_set, char* cpu, DPU_LAUNCH_ARGS *args) {
	dpu_set_t dpu;

	uint32_t idx_dpu = 0;
	size_t element = 0;

	DPU_FOREACH(dpu_set, dpu, idx_dpu) {
		CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu[element])));
		element += args[idx_dpu].num_elements * sizeof(int32_t);
	}

	uint32_t mram_location = args[0].res_offset;
	size_t xfer_size = args[0].num_elements * sizeof(int32_t);
	CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
				DPU_MRAM_HEAP_POINTER_NAME, mram_location,
				xfer_size, DPU_XFER_DEFAULT));
}


int main() {
	// from Param.h
	int32_t elements = nr_elements;
	int nr_of_dpus = dpu_number;

	int iterations = 100;
	dpu_set_t dpu_set;

	CHECK_UPMEM(dpu_alloc(nr_of_dpus, "backend=hw", &dpu_set));
	CHECK_UPMEM(dpu_load(dpu_set, "./bin/baseline.dpu", nullptr));

	DPU_LAUNCH_ARGS args[nr_of_dpus];

	int elements_per_dpu = elements / nr_of_dpus;
	for (uint32_t i = 0; i < nr_of_dpus; i++) {
		args[i].num_elements = elements_per_dpu;
		args[i].lhs_offset = 0;
		args[i].rhs_offset = elements_per_dpu * sizeof(int32_t);
		args[i].res_offset = elements_per_dpu * 2 * sizeof(int32_t);
	}

	// dpu_set_t dpu;
	// uint32_t idx_dpu = 0;
	// DPU_FOREACH(dpu_set, dpu, idx_dpu) {
	//    CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
	//}
	//CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
	//                            sizeof(args[0]), DPU_XFER_DEFAULT));

	int32_t *a_vec = (int32_t*)malloc(elements * sizeof(int32_t));
	int32_t *b_vec = (int32_t*)malloc(elements * sizeof(int32_t));
	int32_t *res_vec = (int32_t*)malloc(elements * sizeof(int32_t));

	vec_xfer_to_dpu(dpu_set, (char*)a_vec, args);
	vec_xfer_to_dpu(dpu_set, (char*)b_vec, args);

	Timer timer;
	start(&timer, 0, 0);

	for (int i = 0; i < iterations; i++) {
		// this is just to simulate libvector dpus sending of arguments each iteration

		int elements_per_dpu = elements / nr_of_dpus;
		for (uint32_t i = 0; i < nr_of_dpus; i++) {
			args[i].num_elements = elements_per_dpu;
			args[i].lhs_offset = 0;
			args[i].rhs_offset = elements_per_dpu * sizeof(int32_t);
			args[i].res_offset = elements_per_dpu * 2 * sizeof(int32_t);
		}

		dpu_set_t dpu;
		uint32_t idx_dpu = 0;
		DPU_FOREACH(dpu_set, dpu, idx_dpu) {
			CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
		}
		CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
					sizeof(args[0]), DPU_XFER_DEFAULT));



		CHECK_UPMEM(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
		// DPU_FOREACH(dpu_set, dpu) { DPU_ASSERT(dpu_log_read(dpu, stdout)); }
	}

	stop(&timer, 0);

	printf("the total time with timing consumed is (ms): ");
	print(&timer, 0, 1);
	printf("\n");

	vec_xfer_from_dpu(dpu_set, (char*)res_vec, args);

	free(a_vec);
	free(b_vec);
	free(res_vec);

	CHECK_UPMEM(dpu_free(dpu_set));

	return 0;
}
