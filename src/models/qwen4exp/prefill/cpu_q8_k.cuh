#pragma once

#include <cstdint>

namespace basalt::prefill::detail {

inline constexpr int kQ8KBlock = 256;
struct CpuQ8K {
    float d;
    int8_t qs[kQ8KBlock];
    int16_t bsums[kQ8KBlock / 16];
};
static_assert(sizeof(CpuQ8K) == 292);

#ifdef __CUDACC__
// Same signed maximum, exact division, rounding and sums as quantize_q8_K_avx2.
static __global__ void quantize_q8k(const float* x, CpuQ8K* out) {
    const int i = threadIdx.x, lane = i % 32;
    const float* row = x + (int64_t) blockIdx.x * kQ8KBlock;
    const float v = row[i];
    const unsigned mag = __float_as_uint(v) & 0x7fffffffu;
    unsigned max_mag = mag;
    for (int step = 16; step > 0; step /= 2)
        max_mag = max(max_mag, __shfl_xor_sync(0xffffffffu, max_mag, step));
    __shared__ unsigned warp_max[8], magnitude;
    __shared__ int first;
    __shared__ float scale;
    if (lane == 0) warp_max[i / 32] = max_mag;
    __syncthreads();
    if (i == 0) {
        magnitude = 0;
        for (int j = 0; j < 8; ++j) magnitude = max(magnitude, warp_max[j]);
        first = kQ8KBlock;
    }
    __syncthreads();
    if (mag == magnitude) atomicMin(&first, i);
    __syncthreads();
    CpuQ8K& dst = out[blockIdx.x];
    if (i == 0) {
        // A nonfinite or overflowed scale is a sentinel for the original CPU path.
        scale = magnitude == 0 || magnitude >= 0x7f800000u ? 0.f : __fdiv_rn(-127.f, row[first]);
        const bool valid = magnitude < 0x7f800000u && isfinite(scale) && (magnitude == 0 || scale != 0.f);
        dst.d = valid ? (scale == 0.f ? 0.f : __fdiv_rn(1.f, scale)) : __int_as_float(0x7fc00000);
        if (!valid) scale = 0.f;
    }
    __syncthreads();
    const int q = scale == 0.f ? 0 : min(127, __float2int_rn(__fmul_rn(scale, v)));
    dst.qs[i] = (int8_t) q;
    int sum = q;
    for (int step = 8; step > 0; step /= 2)
        sum += __shfl_xor_sync(0xffffffffu, sum, step, 16);
    if (i % 16 == 0) dst.bsums[i / 16] = (int16_t) sum;
}
#endif

}  // namespace basalt::prefill::detail
