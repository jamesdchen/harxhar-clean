// puffer_substrate.cuh -- VENDORED from PufferLib 5.0 src/pufferl.cu lines 1-291 (the trainer's
// substrate: precision switch, Float/Prec tensor structs, shape helpers, cast kernels, Allocator,
// block_reduce_sum), commit 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2 (branch 5.0; = release tag
// 5.0-experiments 89414204 + 2 commits, no src/ change).  Upstream src/algo.cu (MinGRU / highway
// kernels, encoder / decoder, cuBLAS wrappers, Muon) is #included UNCHANGED after this file; it was
// written against exactly this substrate (pufferl.cu: 'Algo + sweeps only depend on basic tensor
// types and utilities').  Changes vs upstream lines 1-291 (see substrate.patch):
//   - dropped #include <nccl.h>, <nvml.h>, <nvtx3/nvToolsExt.h> and "ini.h" (RL-trainer only: multi-GPU,
//     GPU utilisation, profiling ranges, config parsing; nothing in algo.cu uses them);
//   - replaced `#include ENV_HEADER` (an RL environment) by the two macros algo.cu's PPO kernels need
//     to compile (NUM_ATNS / ACT_SIZES: one 1-class head; the PPO kernels are compiled, never called).
// CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

// C standard
#include <cassert>
#include <cmath>
#include <cstdint>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// POSIX / threading
#include <dirent.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// Project

#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Compile vs a single env: -DENV_HEADER=ocean/<env>/<env>.h or .cu (--cu)
// (forecast build) no RL environment: algo.cu's PPO kernels only need these two macros to compile.
#define NUM_ATNS 1
#define ACT_SIZES {1}

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} Float;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} Byte;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} Long;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} Int;

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} Prec;

__host__ __device__ int ndim(int64_t* shape) {
    int n = 0;
    while (n < PUF_MAX_DIMS && shape[n] != 0) {
        n++;
    }
    return n;
}

__host__ __device__ int64_t numel(int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) {
        n *= shape[i];
    }
    return n;
}

int64_t batch_size(int64_t* shape) {
    int n = ndim(shape);
    int64_t b = 1;
    for (int i = 0; i < n - 2; i++) {
        b *= shape[i];
    }
    return b;
}

void squeeze_shape(int64_t* shape, int dim) {
    int n = ndim(shape);
    shape[dim] *= shape[dim + 1];
    for (int i = dim + 1; i < n - 1; i++) {
        shape[i] = shape[i + 1];
    }
    shape[n - 1] = 0;
}

Prec* puf_squeeze(Prec* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Float* puf_squeeze(Float* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Prec* puf_unsqueeze(Prec* t, int dim, int64_t d0, int64_t d1) {
    int n = ndim(t->shape);
    assert(n + 1 <= PUF_MAX_DIMS);
    assert(t->shape[dim] == d0 * d1);
    for (int i = n; i > dim; i--) {
        t->shape[i] = t->shape[i - 1];
    }
    t->shape[dim] = d0;
    t->shape[dim + 1] = d1;
    return t;
}

void puf_copy(Prec* dst, Prec* src, cudaStream_t stream) {
    assert(numel(dst->shape) == numel(src->shape) && "puf_copy: size mismatch");
    cudaMemcpyAsync(dst->data, src->data,
        numel(dst->shape) * sizeof(precision_t),
        cudaMemcpyDeviceToDevice, stream);
}

__global__ void cast(precision_t* dst,
        float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(src[idx]);
    }
}

#ifdef PRECISION_FLOAT
// GPU envs fix obs as bf16; float train needs an explicit widen.
__global__ void cast(precision_t* dst,
        __nv_bfloat16* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}
#else
// Identity overload so obs→rollout cast arm typechecks when obs_t is bf16.
__global__ void cast(precision_t* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void cast(float* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}
#endif

__global__ void cast(precision_t* dst,
        unsigned char* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)src[idx]);
    }
}

__global__ void cast(unsigned char* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}

// Fused rew+term cast
__global__ void cast_rew_term(
        precision_t* __restrict__ rew_dst, const float* __restrict__ rew_src,
        precision_t* __restrict__ term_dst, const float* __restrict__ term_src,
        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        rew_dst[idx] = from_float(rew_src[idx]);
        term_dst[idx] = from_float(term_src[idx]);
    }
}

// Allocates large buffers (params/acts/grads) in contiguous memory.
struct AllocEntry {
    void** data_ptr;
    long* shape;
    int elem_size;
};

struct Allocator {
    AllocEntry* regs;
    int num_regs;
    void* mem;
    long total_elems;
    long total_bytes;
};

void _alloc_register(Allocator* alloc,
        void** data_ptr, long* shape, int elem_size) {
    alloc->regs = (AllocEntry*)realloc(alloc->regs,
        (alloc->num_regs + 1) * sizeof(AllocEntry));
    alloc->regs[alloc->num_regs++] = {data_ptr, shape, elem_size};
    long n = numel(shape);
    alloc->total_elems += n;
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += n * elem_size;
}
void alloc_register(Allocator* a, Prec* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(precision_t));
}
void alloc_register(Allocator* a, Float* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(float));
}
void alloc_register(Allocator* a, Long* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(long));
}
void alloc_register(Allocator* a, Int* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(int));
}

void alloc_create(Allocator* alloc) {
    assert(cudaMalloc(&alloc->mem, alloc->total_bytes) == cudaSuccess
        && "alloc_create: cudaMalloc failed");
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *alloc->regs[i].data_ptr = (char*)alloc->mem + offset;
        offset += numel(alloc->regs[i].shape) * alloc->regs[i].elem_size;
    }
}

// Deterministic block tree-reduce. smem layout: smem[c * nthreads + tid].
// Caller fills smem; tid 0 writes nchan results to out[0..nchan).
#ifndef PUF_BLOCK_REDUCE_SUM
#define PUF_BLOCK_REDUCE_SUM
__device__ __forceinline__ void block_reduce_sum(
        float* smem, float* out, int tid, int nthreads, int nchan) {
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + tid] += smem[c * nthreads + tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int c = 0; c < nchan; c++) {
            out[c] = smem[c * nthreads];
        }
    }
}
#endif

