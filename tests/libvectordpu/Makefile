ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif


SRCS := $(wildcard ./*.cc)

INCLUDES := -I../../libs/vectordpu/include/vectordpu
LDFLAGS := -L../../libs/vectordpu/lib
CXXFLAGS := -std=c++20 -O0 $(INCLUDES)

.PHONY: all clean

all: run

run: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)  -Wl,-rpath,../../libs/vectordpu/lib -lvectordpu
clean:
	rm -f run