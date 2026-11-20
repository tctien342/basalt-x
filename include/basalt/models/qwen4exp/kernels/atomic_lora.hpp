#pragma once

#include <cstdint>

namespace basalt::kernels {

/// Add scale * B * dot(A, input) to each contiguous FP32 output row.
/// With experts present, select A using the expert ID for each input/output row.
void atomic_lora_project(const float* input, const float* a, const float* b,
                         const int32_t* experts, int rows, int inputs, int outputs,
                         const float* scale, float* output, void* stream, bool per_row = false);

/// Projection correction for the FP16 activation rows used by prefill GEMM.
void atomic_lora_project_f16(const uint16_t* input, const float* a, const float* b,
                             int rows, int inputs, int outputs, int64_t output_stride,
                             const float* scale, float* output, void* stream, bool per_row = false);

/// Correct the grouped expert down rows in their routed output order. Count and indices are device-resident.
void atomic_lora_grouped_down(const void* scratch, const int32_t* count, const int32_t* dst,
                              const int32_t* ids, int capacity, int inputs, int outputs, const float* a,
                              const float* b, const float* scale, float* output, void* stream, int routes_per_token = 0);

/// Add scale * A[token] * B to each FP32 embedding row. Negative IDs denote external embeddings.
void atomic_lora_embedding(const int32_t* tokens, const float* a, const float* b,
                           int rows, int width, const float* scale, float* output, void* stream, bool per_row = false);
/// The same correction for a host token ID and a single output row.
void atomic_lora_embedding_token(int32_t token, const float* a, const float* b,
                                 int width, const float* scale, float* output, void* stream);

}
