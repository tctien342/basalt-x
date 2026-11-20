#include "basalt/models/qwen4exp/kernels/swift_lora.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int warp_size = 32;
constexpr int warps_per_block = 4;
static_assert(swift_lora_rank == warp_size);

__device__ float activation(float value) { return value; }
__device__ float activation(uint16_t value) { return __half2float(__ushort_as_half(value)); }

__device__ float warp_sum(float value) {
    for (int offset = warp_size / 2; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

template<class Input>
__global__ void swift_lora_reduce(const Input* input, const __nv_bfloat16* a, int inputs,
                                  const float* scale, bool per_row, float* scratch) {
    const int row = blockIdx.y;
    if (scale[per_row ? row : 0] == 0.0f) return;
    const int rank_row = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    input += static_cast<int64_t>(row) * inputs;
    a += static_cast<int64_t>(rank_row) * inputs;
    float sum = 0.0f;
    for (int i = lane; i < inputs; i += warp_size)
        sum = fmaf(activation(input[i]), __bfloat162float(a[i]), sum);
    sum = warp_sum(sum);
    if (lane == 0) scratch[static_cast<int64_t>(row) * swift_lora_rank + rank_row] = sum;
}

__global__ void swift_lora_add(const float* scratch, const __nv_bfloat16* b, int outputs,
                               int64_t output_stride, const float* scale, bool per_row, float* output) {
    const int row = blockIdx.y;
    const float strength = scale[per_row ? row : 0];
    const int column = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
    if (strength == 0.0f || column >= outputs) return;
    const int lane = threadIdx.x % warp_size;
    const float partial = scratch[static_cast<int64_t>(row) * swift_lora_rank + lane] *
                          __bfloat162float(b[static_cast<int64_t>(column) * swift_lora_rank + lane]);
    const float sum = warp_sum(partial);
    if (lane == 0) {
        auto& value = output[static_cast<int64_t>(row) * output_stride + column];
        value = fmaf(strength, sum, value);
    }
}

}

template<class Input>
void swift_lora_project(const Input* input, const uint16_t* a, const uint16_t* b,
                        int rows, int inputs, int outputs, int64_t output_stride,
                        const float* scale, bool per_row, float* scratch,
                        float* output, void* stream) {
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    constexpr int threads = warp_size * warps_per_block;
    swift_lora_reduce<<<dim3(swift_lora_rank / warps_per_block, rows), threads, 0, cuda_stream>>>(
        input, reinterpret_cast<const __nv_bfloat16*>(a), inputs, scale, per_row, scratch);
    swift_lora_add<<<dim3((outputs + warps_per_block - 1) / warps_per_block, rows), threads, 0, cuda_stream>>>(
        scratch, reinterpret_cast<const __nv_bfloat16*>(b), outputs, output_stride, scale, per_row, output);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template void swift_lora_project<float>(
    const float*, const uint16_t*, const uint16_t*, int, int, int, int64_t,
    const float*, bool, float*, float*, void*);
template void swift_lora_project<uint16_t>(
    const uint16_t*, const uint16_t*, const uint16_t*, int, int, int, int64_t,
    const float*, bool, float*, float*, void*);

}
