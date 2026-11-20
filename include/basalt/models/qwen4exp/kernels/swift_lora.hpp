#pragma once

#include <cstdint>

namespace basalt::kernels {

inline constexpr int swift_lora_rank = 32;

// A is [rank, inputs], B is [outputs, rank], both BF16. The caller owns
// scratch [rows, rank] and keeps device addresses stable during graph replay.
template<class Input>
void swift_lora_project(const Input* input, const uint16_t* a, const uint16_t* b,
                        int rows, int inputs, int outputs, int64_t output_stride,
                        const float* scale, bool per_row, float* scratch,
                        float* output, void* stream);

extern template void swift_lora_project<float>(
    const float*, const uint16_t*, const uint16_t*, int, int, int, int64_t,
    const float*, bool, float*, float*, void*);
extern template void swift_lora_project<uint16_t>(
    const uint16_t*, const uint16_t*, const uint16_t*, int, int, int, int64_t,
    const float*, bool, float*, float*, void*);

}
