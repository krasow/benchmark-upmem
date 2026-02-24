ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

NR_TASKLETS ?= 12

BIN_DIR := bin
__dirs := $(shell mkdir -p $(BIN_DIR))

HOST := $(BIN_DIR)/host_baseline
DPU := $(BIN_DIR)/baseline.dpu

all : $(HOST) $(DPU)

# Compiler flags
CFLAGS := -O3 -fopenmp
LIBS := -lm -ldl `dpu-pkg-config --cflags --libs dpu`
DPU_FLAGS := -Wall -Wextra -O3 -DNR_TASKLETS=$(NR_TASKLETS) $(EXTRA_FLAGS)

$(BIN_DIR)/host_baseline: host/baseline.cc host/timer.h Param.h
	g++ $(CFLAGS) host/baseline.cc -o $(BIN_DIR)/host_baseline $(LIBS)

$(BIN_DIR)/baseline.dpu: dpu/dpu_baseline.c Param.h
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $(BIN_DIR)/baseline.dpu dpu/dpu_baseline.c

clean:
	rm -rf $(BIN_DIR)
