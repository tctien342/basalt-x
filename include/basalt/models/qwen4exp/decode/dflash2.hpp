#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::kernels { struct SamplerParams; }
namespace basalt::spec { class DraftPolicy; }

namespace basalt::core {

class NativeHead;
class NativeEmbed;

class Dflash2 {
public:
    static constexpr int kInject = 8;    ///< separate scratch cap for feature projection
    static constexpr int kVerifyRows = 8; ///< maximum target rows captured per verification
    static constexpr int kBlock = 4;      ///< runtime noise block including the anchor
    static constexpr int kDrafts = kBlock - 1;
    static constexpr int kTopK = 16;       ///< selector candidates per position
    static constexpr int kRank = 256;      ///< selector rank
    static constexpr int kLayers = 5;      ///< draft layers
    static constexpr int kHeads = 32, kKvHeads = 8, kHeadDim = 128;
    static constexpr int kMaskToken = 248077;
    static constexpr int kWindow = 2048;   ///< sliding window of every draft layer

    Dflash2() = default;
    ~Dflash2();
    Dflash2(const Dflash2&) = delete;
    Dflash2& operator=(const Dflash2&) = delete;

    // Borrows the target output head and mapped token embedding.
    bool load(const std::string& path, const NativeHead& head, const NativeEmbed* embed,
              int device, std::string& err);
    bool loaded() const { return head_ != nullptr; }

    // The prompt range is request-local. A restored target prefix can start the drafter cold.
    bool begin_prompt(int64_t read_from, int64_t prompt_size, bool live_prefix, std::string& err);
    void capture_prefill(int64_t layer, const float* residual, int rows, int64_t pos, cudaStream_t stream) const;
    void capture_verify(int64_t layer, const float* residual, const float* out, const float* inject,
                        bool pending, int rows, cudaStream_t stream) const;
    bool remember_verify(int64_t pos, int rows, std::string& err);
    bool seed_prompt(int64_t end, std::string& err);
    bool inject_verified(int64_t pos, int rows, std::string& err);
    bool finish(std::string& err);
    bool draft(int32_t anchor, int64_t pos, int max_drafts, const kernels::SamplerParams& sampling,
               const spec::DraftPolicy* cost_policy, int32_t* out, int& n_out, std::string& err,
               void (*after_enqueue)(void*), void* enqueue_context);

private:
    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    const Tensor* find(const std::string& n) const;
    const void* q4(const char* suffix) const;          ///< "fc.weight", "layers.2.self_attn.q_proj.weight"
    const float* f32(const char* suffix) const;
    void inject_rows(const float* rows, int64_t pos, int count);
    void forward_block();
    void enqueue_round(int count, bool capturing = false);
    void mark_profile(int stage, bool capturing = false);
    void launch_round(int count);
    void release();

    int dev_ = -1;
    void* stream_ = nullptr;
    struct RoundIo {
        int32_t tokens[kBlock + 1];  // anchor, mask tokens, absolute position
        int32_t ids[kDrafts * kTopK];
        float vals[kDrafts * kTopK];
        float gates[kDrafts * kRank];
    };
    RoundIo* round_io_ = nullptr;
    cudaGraphExec_t round_graph_ = nullptr;
    bool graph_enabled_ = false;
    bool round_warmed_ = false;
    float max_selector_sum_ = 1.0f / 0.75f;
    bool coupled_proposals_ = false;
    static constexpr int kProfileMarks = 5;
    cudaEvent_t profile_events_[kProfileMarks]{};
    double profile_ms_[kProfileMarks - 1]{};
    uint64_t profile_rounds_ = 0;
    uint64_t profile_proposals_[kDrafts + 1]{};
    uint8_t* dense_ = nullptr;                 // Q4_0/F32 only tensors, offsets from dense.txt
    std::vector<Tensor> tensors_;
    const NativeHead* head_ = nullptr;
    const NativeEmbed* embed_ = nullptr;
    int64_t n_vocab_ = 0;

    // selector codebooks, host: bf16 bits, rows [vocab][rank]
    std::vector<uint16_t> prev_cb_, succ_cb_;
    float cb_prev(int64_t tok, int k) const;   // rank element k of prev[tok]
    float cb_succ(int64_t tok, int k) const;

    // device ring and block workspace (one arena)
    void* arena_ = nullptr;
    uint64_t arena_bytes_ = 0;
    __half* kring_ = nullptr;   // [kLayers][kWindow][kKvHeads][kHeadDim]
    __half* vring_ = nullptr;
    int32_t* kpos_ = nullptr;  // [kLayers][kWindow] token positions (-1 empty)
    float* x_ = nullptr;       // [kBlock][n_embd]
    float* h_ = nullptr;
    float* dyn_ = nullptr;     // [kBlock][2*2*groups] = [kBlock][640]
    float* q_ = nullptr;       // [kBlock][kHeads*kHeadDim]
    float* k_ = nullptr;       // [kInject][kKvHeads*kHeadDim]
    float* v_ = nullptr;
    float* att_part_ = nullptr;  // FP32 softmax partials, shared across layers
    float* att_ = nullptr;     // [kBlock][n_embd]
    float* o_ = nullptr;       // [kBlock][n_embd]
    float* gate_ = nullptr;    // [kBlock][n_ff]
    float* up_ = nullptr;      // [kBlock][n_ff]
    float* y_ = nullptr;       // [kBlock][n_embd]
    float* feat_ = nullptr;    // [kInject][kLayers*n_embd] upload scratch
    float* verify_ = nullptr;  // [kVerifyRows][kLayers*n_embd] graph-stable captured rows
    float* prompt_ = nullptr;  // host staging ring [kWindow][kLayers*n_embd]
    float* prompt_device_ = nullptr;
    int64_t prompt_first_ = 0, prompt_end_ = 0, ring_end_ = -1;
    bool seed_pending_ = false;
    int32_t* tokens_ = nullptr;
    float* g_ = nullptr;       // [n_embd] projected features
    float* logits_ = nullptr;  // [kDrafts][n_vocab]
    void* vhead_scratch_ = nullptr;
    float* gates_ = nullptr;   // [kDrafts][kRank]
    uint8_t* xq_ = nullptr;    // q8_1 scratch
    uint8_t* xq2_ = nullptr;   // q8_1 scratch for 2*n_ff columns
    int32_t* top_tile_ids_ = nullptr;
    float* top_tile_vals_ = nullptr;
    int32_t* top_ids_ = nullptr;   // [kDrafts][kTopK]
    float* top_vals_ = nullptr;
};

// One native engine owns the drafter for the lifetime of its CUDA graphs.
Dflash2* active_dflash2();

}  // namespace basalt::core
