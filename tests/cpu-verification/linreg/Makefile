CXX = g++
CXXFLAGS = -O3 -fopenmp -I../../libvectordpu/linreg/

all: generate_ref

generate_ref: generate_ref.cc ../../libvectordpu/linreg/Param.h
	$(CXX) $(CXXFLAGS) generate_ref.cc -o generate_ref

clean:
	rm -f generate_ref data/*.bin
