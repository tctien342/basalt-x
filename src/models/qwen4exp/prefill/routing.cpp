#include "experts.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/direct_peer.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/prefill/moe_fused.hpp"
#include "basalt/models/qwen4exp/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::moe(int64_t l, const core::LayerView& v) {
    // ======================= MoE =======================
    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                          *wsu = need(v, "ffn_up_shexp.weight", err),
                          *wsd = need(v, "ffn_down_shexp.weight", err);
    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err)) return false;
    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
    if (m.hist_on) route_count(m.ids, T * K, m.hist + l * m.g->n_expert, (int) m.g->n_expert, m.cs);
    if (ahead_n > 0 && l + 1 < LE) {   // the next layer's routing guessed (the logits are free again)
        const core::LayerView vn(*m.wt, l + 1);
        const core::WeightRef* wrn = vn.get("ffn_gate_inp.weight");
        std::string e2;
        if (wrn != nullptr && bf16_proj(m.gemm, wrn, m.mixed_bf, m.logits, T, vn.name("ffn_gate_inp.weight"), e2)) {
            route(m.logits, m.guess_ids, m.guess_w, T, m.g->n_expert, m.cs);
            copy_i32(m.guess_host_dev, m.guess_ids, T * K, m.cs);
            guess_for = l + 1;
        }
    }
    Experts experts(*this, l);
    if (!experts.fused_l && T >= 256 && T <= 4096 && core::exp_flag("route_shared_overlap") != 0) {
        cudaEvent_t ready = nullptr;
        if (cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) == cudaSuccess) {
            Experts::Group group(experts);
            const bool queued = group.read_routes();
            const auto recorded = queued ? cudaEventRecord(ready, m.cs) : cudaSuccess;
            if (recorded != cudaSuccess) err = "prefill: cannot record the route read";
            const bool ok = queued && recorded == cudaSuccess && shared_expert(v, wgi, wsg, wsu, wsd) &&
                            group.execute(ready);
            cudaEventDestroy(ready);
            if (!ok) return false;
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true))
                basalt::log::info("basalt prefill", "route read overlaps shared expert");
            return experts.finish();
        }
        (void) cudaGetLastError();
    }
    if (!shared_expert(v, wgi, wsg, wsu, wsd)) return false;
    return experts.execute();
}

bool Prefill::Run::Chunk::Experts::Group::read_routes() {
    // group the (token, k) pairs by expert on the host
    // (the sync below also orders this layer's writes of slot/src/bounds after the previous
    // layer's kernels that read them)
    if (grp_mapped) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
    else cudaMemcpyAsync(m.ids_host.data(), m.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, m.cs);
    // CPU share (small chunks, experts with a host copy): the layer's activations go to the host on
    // their own stream while the routing is read
    return cpu_download();
}

bool Prefill::Run::Chunk::Experts::Group::group(cudaEvent_t ready) {
    if (!ready && !read_routes()) return false;
    // a stall here is the GPU (this layer's attention and router, or the previous layer's
    // work), not the host: the watchdog's report says so (only its text changes)
    core::progress_at("reading the prompt (batched): waiting for the GPU (attention, router) at layer",
                      l, p0);
    if ((ready ? cudaEventSynchronize(ready) : cudaStreamSynchronize(m.cs)) != cudaSuccess) {
        err = "prefill: cannot synchronize the route read";
        return false;
    }
    core::progress_at("reading the prompt (batched): layer", l, p0);
    if (guess_for == l + 1) {   // the guessed rows per expert of the next layer
        m.guess_rows.assign((size_t) m.g->n_expert, 0);
        for (int64_t i = 0; i < T * K; ++i) {
            const int32_t e = m.guess_host[(size_t) i];
            if (e >= 0 && e < (int32_t) m.g->n_expert) ++m.guess_rows[(size_t) e];
        }
        m.guess_layer = l + 1;
    }
    std::fill(m.cnt.begin(), m.cnt.end(), 0);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t e = ids_h[(size_t) i];
        if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
        ++m.cnt[(size_t) e];
    }
    // multi-GPU: the rows of the experts the peer computes go last, as one block [rows_local, T*K)
    if (ppc && pre_mmq) {
        on_peer.assign((size_t) m.g->n_expert, 0);
        for (int32_t e = 0; e < m.g->n_expert; ++e) {
            const int32_t c = m.cnt[(size_t) e];
            if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                !ppc->peer || !ppc->peer->has(l, e))
                continue;
            if (rows_peer + c > ppc->cap_rows) { continue; }
            on_peer[(size_t) e] = 1;
            rows_peer += c;
        }
        rows_local = T * K - rows_peer;
    }
    cpu_select();
    const bool iq4_cold = mmq_gt == GGML_TYPE_IQ4_XS || mmq_ut == GGML_TYPE_IQ4_XS || mmq_dt == GGML_TYPE_IQ4_XS;
    bool split_peer_result = false;
    const bool cold_peer = !on_peer.empty() && !stream_all && lay.native && T >= 256 && T <= 4096 &&
        core::exp_flag("peer_cold_mmq") != 0 && core::direct_peer_enabled(m.device, ppc->dev);
    if (cold_peer && iq4_cold) {
        // Keep CPU and resident assignments. Avoid moving weights already copied ahead or held on another card.
        std::vector<std::pair<int32_t, int32_t>> cold;   // (rows, expert)
        for (int32_t e = 0; e < m.g->n_expert; ++e) {
            const int32_t c = m.cnt[(size_t) e];
            if (c == 0 || on_peer[(size_t) e] || (!on_cpu.empty() && on_cpu[(size_t) e]) ||
                (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                (ahead_n > 0 && m.ahead_layer[l & 1] == l && m.ahead_slot[l & 1][(size_t) e] >= 0) ||
                !m.src->pinned(l, e) || other_card(e) != nullptr)
                continue;
            cold.emplace_back(c, e);
        }
        std::sort(cold.begin(), cold.end());
        const size_t blob_bytes = lay.blob_bytes(l);
        const bool balance = core::exp_flag("peer_cold_copy_balance") != 0;
        const bool packed = balance && !peer_fp4 && core::exp_flag("peer_result_fp16") != 0 &&
                            core::direct_peer_pointer(m.Dm, m.device);
        const size_t row_bytes = N * (packed ? sizeof(uint16_t) : sizeof(float));
        size_t main_bytes = cold.size() * blob_bytes;
        size_t peer_upload = 0;
        size_t peer_return = rows_peer * row_bytes;
        // Keep the prior conservative byte rule as the default. The optional copy model prices
        // concurrent H2D using measured lanes and charges the peer result at the P2P rate.
        constexpr double main_gbs = 6.66, peer_gbs = 6.92, result_gbs = 5.2;
        for (const auto& [rows, e] : cold) {
            const size_t next_upload = peer_upload + blob_bytes;
            const size_t next_return = peer_return + rows * row_bytes;
            if (rows_peer + rows > ppc->cap_rows) break;
            if (balance) {
                const double before = std::max(main_bytes / main_gbs,
                                               peer_upload / peer_gbs + peer_return / result_gbs);
                const double after = std::max((main_bytes - blob_bytes) / main_gbs,
                                              next_upload / peer_gbs + next_return / result_gbs);
                if (after >= before) break;
            } else if (next_upload + next_return >= main_bytes) {
                break;
            }
            on_peer[(size_t) e] = 1;
            rows_peer += rows;
            rows_local -= rows;
            main_bytes -= blob_bytes;
            peer_upload = next_upload;
            peer_return = next_return;
        }
        if (balance && peer_upload) {
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true))
                basalt::log::info("basalt prefill", "cold copy balance moved %zu experts; estimated peer return %zu bytes",
                                  peer_upload / blob_bytes, peer_return);
        }
    } else if (cold_peer && ppc->peer && core::exp_flag("peer_cold_mmq_ranked") != 0) {
        // Keep the exact primary MMQ group boundaries. Stream-K may sum a row
        // differently when the group size or largest expert row count changes.
        std::vector<int32_t> local;
        const auto streams_on_main = [&](int32_t e) {
            const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
            const bool ahead = ahead_n > 0 && m.ahead_layer[l & 1] == l && m.ahead_slot[l & 1][(size_t) e] >= 0;
            return !resident && !ahead && other_card(e) == nullptr;
        };
        const size_t blob_bytes = lay.blob_bytes(l);
        constexpr double main_gbs = 6.66, peer_gbs = 6.92, result_gbs = 5.2;
        const bool packed = !peer_fp4 && core::exp_flag("peer_result_fp16") != 0 &&
                            core::direct_peer_pointer(m.Dm, m.device);
        const size_t old_row_bytes = N * (packed ? sizeof(uint16_t) : sizeof(float));
        size_t primary_bytes = 0;
        double peer_cost = (double) rows_peer * old_row_bytes / result_gbs;
        for (int32_t e = 0; e < m.g->n_expert; ++e) {
            if (m.cnt[(size_t) e] == 0 || on_peer[(size_t) e] || (!on_cpu.empty() && on_cpu[(size_t) e])) continue;
            local.push_back(e);
            if (streams_on_main(e)) primary_bytes += blob_bytes;
        }
        struct Group {
            size_t begin, end, saved, upload;
            int64_t rows;
            double cost;
        };
        std::vector<Group> eligible;
        for (size_t begin = 0; begin < local.size(); begin += MMQ_GROUP) {
            const size_t end = std::min(local.size(), begin + MMQ_GROUP);
            size_t saved = 0, upload = 0;
            int64_t rows = 0;
            bool available = true;
            for (size_t j = begin; j < end; ++j) {
                const int32_t e = local[j];
                if (!ppc->peer->has(l, e)) {
                    if (!m.src->pinned(l, e)) { available = false; break; }
                    upload += blob_bytes;
                }
                if (streams_on_main(e)) saved += blob_bytes;
                rows += m.cnt[(size_t) e];
            }
            if (!available || saved == 0 || saved * 2 < upload || rows > ppc->G) continue;
            const double cost = upload / peer_gbs + (double) rows * N * sizeof(float) / result_gbs;
            eligible.push_back({begin, end, saved, upload, rows, cost});
        }
        std::stable_sort(eligible.begin(), eligible.end(), [](const Group& a, const Group& b) {
            return a.saved != b.saved ? a.saved > b.saved : a.cost < b.cost;
        });
        size_t moved = 0, saved_bytes = 0, uploaded_bytes = 0;
        for (const Group& group : eligible) {
            if (rows_peer + group.rows > ppc->cap_rows) continue;
            const double peer_next = peer_cost + group.cost;
            if (std::max((primary_bytes - group.saved) / main_gbs, peer_next) >=
                std::max(primary_bytes / main_gbs, peer_cost)) continue;
            // Preserve expert order within each group and keep the primary's FP32 result precision.
            for (size_t j = group.begin; j < group.end; ++j) on_peer[(size_t) local[j]] = 2;
            ++moved;
            saved_bytes += group.saved;
            uploaded_bytes += group.upload;
            primary_bytes -= group.saved;
            peer_cost = peer_next;
            rows_peer += group.rows;
            rows_local -= group.rows;
        }
        split_peer_result = moved > 0;
        if (split_peer_result) {
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true))
                basalt::log::info("basalt prefill",
                                  "cold peer MMQ ranked and moved %zu whole groups (%zu primary bytes avoided, %zu peer uploads)",
                                  moved, saved_bytes, uploaded_bytes);
        }
    }
    {
        int32_t r = 0;
        for (int32_t e = 0; e < m.g->n_expert; ++e)
            if ((on_peer.empty() || !on_peer[(size_t) e]) && (on_cpu.empty() || !on_cpu[(size_t) e])) {
                m.off[(size_t) e] = r;
                r += m.cnt[(size_t) e];
            }
        // The peer's FP32 prefix precedes its original FP16 tail. Keep each range in expert order.
        for (int mode = split_peer_result ? 2 : 1; mode > 0 && !on_peer.empty(); --mode)
            for (int32_t e = 0; e < m.g->n_expert; ++e)
                if (on_peer[(size_t) e] == mode) {
                    m.off[(size_t) e] = r;
                    r += m.cnt[(size_t) e];
                    order_peer.push_back(e);
                }
        for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block, last
            if (on_cpu[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
        m.off[(size_t) m.g->n_expert] = r;
    }
    std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t e = ids_h[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot_h[(size_t) i] = p;
        src_h[(size_t) p] = (int32_t) (i / K);
    }
    if (grp_mapped) {
        copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
        copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
    } else {
        cudaMemcpyAsync(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
        cudaMemcpyAsync(m.src_dev, m.src_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
    }
    // the experts, in id order: resident ones from VRAM, the others through the staging ring
    for (int32_t e = 0; e < m.g->n_expert; ++e)
        if (m.cnt[(size_t) e] > 0 && (on_cpu.empty() || !on_cpu[(size_t) e]) &&
            (on_peer.empty() || !on_peer[(size_t) e]))
            order.push_back(e);
    // CPU share: the pool runs the CPU's experts while this thread stages and runs the GPU's; their
    return true;
}

}  // namespace basalt::prefill
