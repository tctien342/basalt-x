#include "state.hpp"
#include "basalt/models/qwen4exp/atomic_lora.hpp"
#include "basalt/models/qwen4exp/swift_lora.hpp"
#include "basalt/models/qwen4exp/kernels/swift_lora.hpp"
#include "basalt/models/qwen4exp/kernels/atomic_lora.hpp"
#include "layout.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <exception>
#include <string>

namespace basalt::prefill {
using namespace detail;

namespace detail {
// The hc up product and the mix: one kernel from kUpMixMinT tokens up (the gate rows never reach memory), else the
// product into `gated` and gr_mix_r.  The same bits either way.
void up_mix(Gemm& gm, const uint16_t* w_up, const uint16_t* lo16, float* gated, const float* R, const float* rs,
            const float* w_norm, float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream) {
    const uint16_t* w = T >= kUpMixMinT ? gm.bf16_rows(w_up, D, LR) : nullptr;
    if (w != nullptr && gr_up_mix(lo16, w, R, rs, w_norm, mixed, mixed16, T, stream, mixed_h)) return;
    gm.bf16_w(lo16, w_up, gated, T, D, LR);
    gr_mix_r(R, rs, w_norm, gated, mixed, mixed16, T, stream, mixed_h);
}

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (the model pack must carry them)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    if (const auto* adapter = gm.lora()) {
        if (const float* a = adapter->find_input_factor(name)) {
            try {
                basalt::kernels::atomic_lora_project_f16(X, a, adapter->output_factor(), (int) T,
                    (int) w->ne0, (int) w->ne1, ldy ? ldy : w->ne1, adapter->scale(), Y, gm.stream());
            } catch (const std::exception& error) {
                err = "prefill LoRA: " + name + ": " + error.what();
                return false;
            }
        }
    }
    if (const auto* adapter = gm.swift_lora()) {
        if (const auto* factors = adapter->find(name)) {
            constexpr auto rank = basalt::kernels::swift_lora_rank;
            if (T > 65535 || T > gm.scratch_elems() / (2 * rank)) {
                err = "prefill Swift LoRA: insufficient projection scratch for " + name;
                return false;
            }
            try {
                // The base product and delta share the same stream. The dequantized
                // weight has no consumer after the base GEMM.
                basalt::kernels::swift_lora_project(X, factors->a, factors->b,
                    (int) T, (int) w->ne0, (int) w->ne1, ldy ? ldy : w->ne1,
                    adapter->scale(), false, reinterpret_cast<float*>(gm.scratch()), Y, gm.stream());
            } catch (const std::exception& error) {
                err = "prefill Swift LoRA: " + name + ": " + error.what();
                return false;
            }
        }
    }
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16_w(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}

}  // namespace basalt::prefill
