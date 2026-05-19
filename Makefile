# Makefile — FLUX GPU micro-experiments
# RTX 4050 (SM 8.9), CUDA 12.6

CUDA_PATH ?= /usr/local/cuda-12.6
NVCC      := $(CUDA_PATH)/bin/nvcc
ARCH      := sm_89
OPT       := -O3
FLAGS     := -arch=$(ARCH) $(OPT) --expt-relaxed-constexpr -lineinfo

# Ensure CUDA libs are findable
LDFLAGS   := -L$(CUDA_PATH)/lib64 -lcudart

TARGETS   := exact_check batch_check sediment bfs hyperbolic

.PHONY: all bench clean

all: $(TARGETS)

exact_check: exact_check_kernel.cu
	$(NVCC) $(FLAGS) $< -o $@ $(LDFLAGS)

batch_check: batch_check_kernel.cu
	$(NVCC) $(FLAGS) $< -o $@ $(LDFLAGS)

sediment: sediment_kernel.cu
	$(NVCC) $(FLAGS) $< -o $@ $(LDFLAGS)

bfs: bfs_kernel.cu
	$(NVCC) $(FLAGS) $< -o $@ $(LDFLAGS)

hyperbolic: hyperbolic_kernel.cu
	$(NVCC) $(FLAGS) $< -o $@ $(LDFLAGS)

bench: all
	@echo "Running benchmarks..."
	@bash run_all_benchmarks.sh

clean:
	rm -f $(TARGETS) *.o
