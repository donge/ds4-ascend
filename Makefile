CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CORE_OBJS = ds4.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression ascend ascend-smoke

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, and ./ds4-bench"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, and ./ds4-bench"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make ascend              Build AscendCL backend for Atlas / Ascend NPU"
	@echo "  make ascend-smoke        Build and run AscendCL kernel smoke test"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, and ./ds4-bench"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) ds4 ds4-server ds4-bench CUDA_ARCH=

cuda-generic:
	$(MAKE) ds4 ds4-server ds4-bench CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) ds4 ds4-server ds4-bench CUDA_ARCH="$(CUDA_ARCH)"

ASCEND_HOME ?= /usr/local/Ascend/cann-9.0.0
ASCEND_CFLAGS ?= -DDS4_ASCEND_BACKEND -I$(ASCEND_HOME)/aarch64-linux/include
ASCEND_DRIVER_LIB ?= /usr/local/Ascend/driver/lib64/driver
ASCEND_LDLIBS ?= $(LDLIBS) -L$(ASCEND_HOME)/aarch64-linux/lib64 -L$(ASCEND_DRIVER_LIB) -Wl,-rpath,$(ASCEND_DRIVER_LIB) -lruntime -lascendcl -lascend_hal -lstdc++
ASCENDC ?= $(ASCEND_HOME)/aarch64-linux/bin/ccec
ASCENDCFLAGS ?= -x cce --cce-aicore-arch=dav-m200 -I$(ASCEND_HOME)/aarch64-linux/include -I$(ASCEND_HOME)/aarch64-linux/include/experiment -I$(ASCEND_HOME)/aarch64-linux/asc -I$(ASCEND_HOME)/aarch64-linux/asc/include -I$(ASCEND_HOME)/aarch64-linux/asc/include/basic_api -I$(ASCEND_HOME)/aarch64-linux/asc/include/interface -I$(ASCEND_HOME)/aarch64-linux/asc/impl/basic_api -I$(ASCEND_HOME)/aarch64-linux/ascendc/include -I$(ASCEND_HOME)/aarch64-linux/ascendc/include/basic_api -I$(ASCEND_HOME)/aarch64-linux/ascendc/include/basic_api/impl -I$(ASCEND_HOME)/aarch64-linux/ascendc/include/highlevel_api -O2 -std=c++17 -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-fp-ceiling=2 -mllvm -cce-aicore-record-overflow=false
ASCEND_CORE_OBJS = ds4_ascend_build.o ds4_ascend_backend.o ds4_ascend_kernels.o

ascend: ds4_ascend ds4-server-ascend ds4-bench-ascend

ascend-smoke: tests/ascend_fill_smoke
	./tests/ascend_fill_smoke

# Keep the normal CUDA object names untouched; Ascend builds use separate names
# so developers can switch between CUDA and Ascend without stale object reuse.
ds4_cli_ascend.o linenoise.o $(ASCEND_CORE_OBJS): CFLAGS += $(ASCEND_CFLAGS) -fPIC

ds4_server_ascend.o rax.o: CFLAGS += $(ASCEND_CFLAGS) -fPIC

ds4_bench_ascend.o: CFLAGS += $(ASCEND_CFLAGS) -fPIC

ds4_ascend: ds4_cli_ascend.o linenoise.o $(ASCEND_CORE_OBJS)
	$(ASCENDC) -o $@ $^ $(ASCEND_LDLIBS)

ds4-server-ascend: ds4_server_ascend.o rax.o $(ASCEND_CORE_OBJS)
	$(ASCENDC) -o $@ $^ $(ASCEND_LDLIBS)

ds4-bench-ascend: ds4_bench_ascend.o $(ASCEND_CORE_OBJS)
	$(ASCENDC) -o $@ $^ $(ASCEND_LDLIBS)

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ascend_build.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_cli_ascend.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_server_ascend.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_bench_ascend.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

tests/ascend_fill_smoke.o: CFLAGS += $(ASCEND_CFLAGS) -fPIC

tests/ascend_fill_smoke.o: tests/ascend_fill_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/ascend_fill_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_ascend_backend.o: ds4_ascend.c ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ascend.c

ds4_ascend_kernels.o: ds4_ascend_kernels.cpp
	$(ASCENDC) -c $(ASCENDCFLAGS) -o $@ ds4_ascend_kernels.cpp

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/ascend_fill_smoke: tests/ascend_fill_smoke.o ds4_ascend_backend.o ds4_ascend_kernels.o
	$(ASCENDC) -o $@ tests/ascend_fill_smoke.o ds4_ascend_backend.o ds4_ascend_kernels.o $(ASCEND_LDLIBS)

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4_ascend ds4-server-ascend ds4-bench-ascend ds4_ascend_kernels.bin ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/ascend_fill_smoke tests/ascend_fill_smoke.o
