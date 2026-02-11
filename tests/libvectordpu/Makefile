ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif


SRCS := $(wildcard ./*.cc)

INCLUDES := -I../../opt/vectordpu/include/vectordpu
LDFLAGS := -L../../opt/vectordpu/lib
CXXFLAGS := -std=c++17 -O3 $(INCLUDES) -g -pg -fno-omit-frame-pointer

.PHONY: all clean

all: run

run: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)  -Wl,-rpath,../../opt/vectordpu/lib -lvectordpu
clean:
	rm -f run
