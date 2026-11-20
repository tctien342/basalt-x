#include "basalt/core/direct_peer.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"
#include <algorithm>
#include "basalt/log.hpp"
#include <atomic>
#include <chrono>
#include "run.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/peer_experts.hpp"
#include <string>
#include <future>

namespace basalt::prefill {
namespace {

cudaError_t copy_packed_rows(const float* src, void* dst, int64_t count, float* scratch, int64_t capacity,
                             int src_device, int dst_device, cudaStream_t stream) {
    auto* packed = reinterpret_cast<uint16_t*>(scratch);
    auto* out = static_cast<uint16_t*>(dst);
    for (int64_t at = 0; at < count; at += capacity) {
        const int64_t n = std::min(capacity, count - at);
        to_f16(src + at, packed, n, stream);
        const size_t bytes = (size_t) n * sizeof(uint16_t);
        const auto status = dst_device < 0
            ? cudaMemcpyAsync(out + at, packed, bytes, cudaMemcpyDeviceToHost, stream)
            : cudaMemcpyPeerAsync(out + at, dst_device, packed, src_device, bytes, stream);
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

}  // namespace

bool Prefill::copy_draft_rows_fp16(const float* rows, void* host, int64_t count, std::string& err) {
    auto& m = *impl_;
    const core::OnDevice on(m.device);
    const auto status = copy_packed_rows(rows, host, count, m.mixed, m.T * m.g->n_embd * 2, m.device, -1, m.cs);
    if (status == cudaSuccess && cudaStreamSynchronize(m.cs) == cudaSuccess) return true;
    err = "prefill: FP16 draft relay copy failed";
    return false;
}
using namespace detail;

int Prefill::Run::Chunk::finish() {
    if (!ple_land()) return -1;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
    if (issuer.joinable()) {
        issuer.join();
        owner.stats_.ms_experts_host += iss_ms;
        owner.stats_.experts_streamed += iss_streamed;
        owner.stats_.experts_dma += iss_dma;
    }
    owner.stats_.tokens += T;
    core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
    if (owner.next_ != nullptr && owner.keep_last_ && c0 + T >= n) return 0;   // kept: its later layers run on this card next
    if (owner.next_ != nullptr && owner.keep_last_ && owner.keep_here && owner.keep_here(p0 + T)) return 1;   // kept, and the run ends with it
    if (owner.next_ != nullptr) {
        if (T > m.hand_rows || m.hand[owner.hand_buf_] == nullptr) {
            err = "prefill: the layer split's hand-off buffers are too small for the chunk";
            return -1;
        }
        // The current hand-off slot was used two chunks ago.
        float* h = m.hand[owner.hand_buf_];
        auto& next = *owner.next_->impl_;
        const size_t bytes = (size_t) T * D * 4;
        const bool peer_ok = T <= next.T && core::direct_peer_enabled(next.device, m.device) &&
                            core::direct_peer_pointer(next.R, next.device, m.device, bytes) &&
                            core::direct_peer_pointer(m.R, m.device, next.device, bytes);
        // A busy successor still owns R. The existing host slot can start draining the producer immediately.
        const bool busy = peer_ok && core::exp_flag("activation_overlap") != 0 && owner.next_run_.valid() &&
                          owner.next_run_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
        const bool direct = peer_ok && !busy;
        if (busy) {
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true))
                basalt::log::info("basalt P2P", "layer activation staged while successor still owns destination");
        }
        // The destination belongs to the successor until its previous chunk finishes.
        if (direct && owner.next_run_.valid() && !owner.next_run_.get()) { err = owner.next_err_; return -1; }
        const bool packed = direct && core::exp_flag("handoff_fp16") != 0;
        const auto copy = packed ? copy_packed_rows(m.R, next.R, T * D, m.mixed, m.T * m.g->n_embd * 2, m.device, next.device, m.cs)
                        : direct ? cudaMemcpyPeerAsync(next.R, next.device, m.R, m.device, bytes, m.cs)
                                 : cudaMemcpyAsync(h, m.R, bytes, cudaMemcpyDeviceToHost, m.cs);
        if (copy != cudaSuccess || cudaStreamSynchronize(m.cs) != cudaSuccess) {
            err = std::string("prefill: the layer split's hand-off: ") + cudaGetErrorString(cudaGetLastError());
            return -1;
        }
        if (packed) {
            const core::OnDevice on(next.device);
            expand_f16_in_place(next.R, T * D, next.cs);
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true))
                basalt::log::info("basalt P2P", "layer activation FP16 CUDA%d -> CUDA%d (%zu bytes)",
                                  m.device, next.device, bytes / 2);
        }
        if (direct && !packed) {
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true)) basalt::log::info("basalt P2P", "layer activation CUDA%d -> CUDA%d (%zu bytes)",
                                                          m.device, next.device, bytes);
        }
        if (owner.on_stage_chunk && !owner.on_stage_chunk(p0 + T, err)) return -1;
        if (owner.next_run_.valid() && !owner.next_run_.get()) { err = owner.next_err_; return -1; }
        owner.next_err_.clear();
        owner.next_->hand_in_ = h;   // the host slot remains the MTP relay destination
        owner.next_->hand_on_device_ = direct;
        owner.next_->single_chunk_ = single_chunk;
        owner.next_run_ = std::async(std::launch::async,
            [&owner = owner, tokens = tokens, c0 = c0, T = T, p0 = p0, context = kernels::rope_context()] {
                const kernels::RopeContextScope scope(context);
                return owner.next_->run_impl(tokens + c0, T, p0, owner.next_err_);
            });
        owner.hand_buf_ ^= 1;
        return 0;   // the last stage reports the chunk (on_chunk)
    }
    if (owner.on_chunk || owner.on_stage_chunk) {
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
            err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
            return -1;
        }
        if (owner.on_stage_chunk && !owner.on_stage_chunk(p0 + T, err)) return -1;
        if (owner.on_chunk && !owner.on_chunk(m.R, T, p0, err)) return -1;
    }
    return 0;
}

}  // namespace basalt::prefill
