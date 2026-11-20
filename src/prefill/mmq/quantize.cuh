// SPDX-License-Identifier: MIT
// llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d (https://github.com/ggml-org/llama.cpp),
// ggml/src/ggml-cuda/quantize.cuh.  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#pragma once

#include "mmq.cuh"
#include "mmq_common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

// Basalt's expert activation, shared by the float output and fused MMQ loaders.
struct MmqSwiglu {
    bool interleaved;
    const float4* tail;
    const int32_t* bounds;
    int n;
};

static __device__ __forceinline__ const float4* mmq_swiglu_scales(MmqSwiglu act, int64_t row) {
    if (act.tail == nullptr) return nullptr;
    int q = 0;
    while (q + 1 < act.n && act.bounds[q + 1] <= row) ++q;
    return act.tail + q;
}

static __device__ __forceinline__ float mmq_swiglu_value(const float* row, int64_t k, int64_t n_ff,
                                                       bool interleaved, const float4* scale) {
    float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    if (scale == nullptr) return g / (1.0f + __expf(-g)) * u;
    const float4 s = *scale;
    g *= s.x;
    u *= s.y;
    return g / (1.0f + __expf(-g)) * u * s.z;
}

void quantize_mmq_swiglu_q8_1_cuda(const float* gu, void* vy, ggml_type type_x, int64_t n_ff, int64_t padded_cols,
                                  int64_t rows, MmqSwiglu act, cudaStream_t stream);

// Quantizes `ne1` rows of `ne0` floats (row i taken from x + ids[i] * s01 when `ids` is given) to block_q8_1_mmq,
// in the scale layout the weights of `type_x` multiply against.
void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, ggml_type type_x,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);
