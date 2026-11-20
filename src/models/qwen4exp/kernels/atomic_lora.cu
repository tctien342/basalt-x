#include "basalt/models/qwen4exp/kernels/atomic_lora.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int kThreads = 128;

__device__ float activation(float value) { return value; }
__device__ float activation(__half value) { return __half2float(value); }

template<class Input>
__global__ void rank_one_project(const Input* input, const float* a, const float* b,
                                 const int32_t* experts, int inputs, int outputs, int64_t output_stride,
                                 const float* scale, float* output, bool per_row) {
    const size_t row = blockIdx.x;
    const float strength = scale[per_row ? row : 0];
    if (strength == 0) return;
    const int expert = experts ? experts[row] : 0;
    if (expert < 0) return;
    input += row * inputs;
    a += static_cast<size_t>(expert) * inputs;
    output += row * output_stride;
    __shared__ float partial[kThreads];
    float dot = 0;
    for (int i = threadIdx.x; i < inputs; i += kThreads)
        dot = fmaf(activation(input[i]), a[i], dot);
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float gain = strength * partial[0];
    for (int i = threadIdx.x; i < outputs; i += kThreads)
        output[i] = fmaf(gain, b[i], output[i]);
}

__global__ void rank_one_grouped(const float* gate, const float* up, const int32_t* count,
                                 const int32_t* dst, const int32_t* ids, int inputs, int outputs,
                                 const float* a, const float* b, const float* scale, float* output, int routes_per_token) {
    if (blockIdx.x >= (unsigned) *count) return;
    const size_t row = blockIdx.x;
    const int route = dst[row];
    const float strength = scale[routes_per_token > 0 ? route / routes_per_token : 0];
    if (strength == 0) return;
    const int expert = ids[route];
    if (expert < 0) return;
    gate += row * inputs;
    up += row * inputs;
    a += (size_t) expert * inputs;
    output += (size_t) route * outputs;
    __shared__ float partial[kThreads];
    float dot = 0;
    for (int i = threadIdx.x; i < inputs; i += kThreads) {
        const float g = gate[i];
        const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]);
        dot = fmaf(h, a[i], dot);
    }
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float gain = strength * partial[0];
    for (int i = threadIdx.x; i < outputs; i += kThreads)
        output[i] = fmaf(gain, b[i], output[i]);
}

__global__ void rank_one_embedding(const int32_t* tokens, const float* a, const float* b,
                                   int32_t host_token, int width, const float* scale, float* output, bool per_row) {
    const size_t row = blockIdx.x;
    const float strength = scale[per_row ? row : 0];
    if (strength == 0) return;
    const int token = tokens ? tokens[row] : host_token;
    if (token < 0) return;
    const float gain = strength * a[token];
    output += row * width;
    for (int i = threadIdx.x; i < width; i += kThreads)
        output[i] = fmaf(gain, b[i], output[i]);
}

void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}

void atomic_lora_project(const float* input, const float* a, const float* b,
                         const int32_t* experts, int rows, int inputs, int outputs,
                         const float* scale, float* output, void* stream, bool per_row) {
    if (rows == 0 || scale == nullptr) return;
    if (!input || !a || !b || !output || rows < 0 || inputs < 1 || outputs < 1)
        throw std::invalid_argument("AtomicChat LoRA: invalid projection rows");
    rank_one_project<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        input, a, b, experts, inputs, outputs, outputs, scale, output, per_row);
    check_launch();
}

void atomic_lora_project_f16(const uint16_t* input, const float* a, const float* b,
                             int rows, int inputs, int outputs, int64_t output_stride,
                             const float* scale, float* output, void* stream, bool per_row) {
    if (rows == 0 || scale == nullptr) return;
    if (!input || !a || !b || !output || rows < 0 || inputs < 1 || outputs < 1 || output_stride < outputs)
        throw std::invalid_argument("AtomicChat LoRA: invalid FP16 projection rows");
    rank_one_project<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __half*>(input), a, b, nullptr, inputs, outputs, output_stride, scale, output, per_row);
    check_launch();
}

void atomic_lora_grouped_down(const void* scratch, const int32_t* count, const int32_t* dst,
                              const int32_t* ids, int capacity, int inputs, int outputs, const float* a,
                              const float* b, const float* scale, float* output, void* stream, int routes_per_token) {
    if (!scale || capacity == 0) return;
    if (!scratch || !count || !dst || !ids || !a || !b || !output || capacity < 0 || inputs < 1 || outputs < 1)
        throw std::invalid_argument("AtomicChat LoRA: invalid grouped projection");
    const size_t offset = ((size_t) capacity * inputs * sizeof(float) + 255) & ~(size_t) 255;
    rank_one_grouped<<<capacity, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(scratch), reinterpret_cast<const float*>(static_cast<const uint8_t*>(scratch) + offset),
        count, dst, ids, inputs, outputs, a, b, scale, output, routes_per_token);
    check_launch();
}

void atomic_lora_embedding(const int32_t* tokens, const float* a, const float* b,
                           int rows, int width, const float* scale, float* output, void* stream, bool per_row) {
    if (rows == 0 || scale == nullptr) return;
    if (!tokens || !a || !b || !output || rows < 0 || width < 1)
        throw std::invalid_argument("AtomicChat LoRA: invalid embedding rows");
    rank_one_embedding<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        tokens, a, b, -1, width, scale, output, per_row);
    check_launch();
}

void atomic_lora_embedding_token(int32_t token, const float* a, const float* b,
                                 int width, const float* scale, float* output, void* stream) {
    if (scale == nullptr || token < 0) return;
    if (!a || !b || !output || width < 1)
        throw std::invalid_argument("AtomicChat LoRA: invalid embedding row");
    rank_one_embedding<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        nullptr, a, b, token, width, scale, output, false);
    check_launch();
}

}
