CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse
C_FLAGS      := -Wall -Wextra -Wno-deprecated-declarations -fPIC -g
FLAGS        := --extended-lambda --expt-relaxed-constexpr -O3 -G $(addprefix -Xcompiler ,$(C_FLAGS))

FP_TYPE ?= -DFP64

all: clean spmv spmm

spmv:
	nvcc $(INC) $(FLAGS) $(FP_TYPE) spmv_csr_example.cpp -o spmv $(LIBS)

spmm:
	nvcc $(INC) $(FLAGS) $(FP_TYPE) spmm_csr_example.cpp -o spmm $(LIBS)

clean:
	rm -rf spmv spmm
