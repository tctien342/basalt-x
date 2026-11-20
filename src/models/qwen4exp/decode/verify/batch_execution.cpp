#include "common.hpp"
#include "expert_plan.hpp"
#include "diagnostics.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/models/qwen4exp/kernels/dec_gdn.hpp"
#include "basalt/models/qwen4exp/kernels/native_ple_postops.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/kernels/wait_abort.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <immintrin.h>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::stage_batch(const int* slots, const int* lens, int S, int hbase, const int32_t* tokens,
                           const int64_t* pos0, std::string& err) {
    using namespace basalt::kernels;
    int R = 0;
    for (int s = 0; s < S && s < 9; ++s) R += lens != nullptr ? lens[s] : 1;
    if (S < 1 || S > max_t_ || R < S || R > max_t_ || hbase < 0 || (hbase > 0 && hbase + R > (int) slots_.size())) {
        err = "verify: batch rows out of range (init_slots)";
        return false;
    }
    for (int s = 0; s < S; ++s) {
        bool dup = false;
        for (int u = 0; u < s; ++u) dup = dup || slots[u] == slots[s];
        if (slots[s] < 0 || slots[s] >= (int) slots_.size() || dup || (lens != nullptr && lens[s] < 1)) {
            err = "verify: a batch segment's slot is out of range or twice, or its length is empty";
            return false;
        }
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU; restart the engine"; return false; }
    if (basalt::kernels::wait_abort_code() != 0) { err = "verify: a device wait timed out in an earlier window; restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    for (int s = 0; s < S; ++s)
        if (pos0[s] < 0 || pos0[s] + (lens != nullptr ? lens[s] : 1) > slots_[(size_t) slots[s]]->max_cells) {
            err = "verify: slot " + std::to_string(slots[s]) + " runs past its context";
            return false;
        }
    refresh_ar();
    swift_batch_window_ = false;
    if (swift_lora_)
        for (int s = 0; s < S; ++s)
            swift_batch_window_ |= slot_swift_scale_[(size_t) slots[s]] != 0.0f;
    if (!capture_batch(R, hbase, err) || !capture_commit_batch(R, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    const int64_t CB = 2 + max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const bool do_ple = ss_->ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < kQsaSegMaxRows + 8; ++t) h_qseg_[t] = QsaSeg{-1, 0, 0, 0};   // unused until set
    const int64_t nQ = g.n_qsa_layers();
    const int64_t TS = (s.idx_block - 1) * g.idx_key_dim;
    const int64_t interval = std::max<int64_t>(g.qsa_interval, 1);
    for (int64_t qi = lb_ / interval; qi < le_ / interval; ++qi)
        for (int sg = 0; sg < S; ++sg) {
            const QsaState& st = slots_[(size_t) slots[sg]]->qsa_states[qi];
            QsaSegSlot& e = h_qslots_[(size_t) qi * max_t_ + sg];
            e.k_pool = st.k_pool; e.v_pool = st.v_pool; e.k_q = st.k_q; e.v_q = st.v_q;
            e.k_scale = st.k_scale; e.v_scale = st.v_scale; e.page_table = st.page_table; e.host = st.host;
            e.map = st.kv_mode == 1 ? st.map : decltype(e.map){};
            e.tail = st.idx_tail; e.dead = st.idx_dead; e.pooled = st.idx_pooled;
            e.tail_snap = tail_snap_b_ + ((size_t) sg * nQ + qi) * TS;
            e.block_pos = st.idx_block_pos; e.max_cells = (int32_t) st.max_cells;
        }
    for (int sg = 0, r0 = 0; sg < S; ++sg) {
        const int n = lens != nullptr ? lens[sg] : 1;
        const SessionState& sx = *slots_[(size_t) slots[sg]];
        h_ple_slots_[sg] = sx.ple_hist;
        int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
        h_qseg_[kQsaSegMaxRows + sg] = QsaSeg{sg, r0, n, 0};
        int32_t* c = h_commitb_ + (size_t) sg * CB;   // the commit's rows, all kept until commit_slots says less
        c[0] = n;
        c[1] = n - 1;
        for (int64_t j = 0; j < CB - 2; ++j) c[2 + j] = j < n ? (int32_t) (pos0[sg] + j) : -1;
        for (int j = 0; j < n; ++j) {
            const int t = r0 + j;
            const int64_t pos = pos0[sg] + j;
            h_tok_[t] = tokens[t];
            h_adapter_scale_[t] = slot_adapter_scale_[(size_t) slots[sg]];
            if (h_swift_scale_) h_swift_scale_[t] = slot_swift_scale_[(size_t) slots[sg]];
            qsa_step_fill(h_step_ + t * kStepCount, pos, s);
            for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) pos;
            for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) pos;
            for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) pos;
            if (do_ple) {
                ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + t * PLE_N_HEADS);
                prev[0] = prev[1];
                prev[1] = tokens[t];
            }
            h_qseg_[t] = QsaSeg{sg, r0, n, 0};
            SegRow& e = h_segb_[t];   // the state of layer 0 of the slot's session
            e.gdn = (float*) ((uintptr_t) sx.gdn_state - (uintptr_t) sx.gdn_ord0 * gdn_floats * sizeof(float));
            e.row0 = r0; e.j = j; e.n = n; e.n_keep = n; e.segment = sg; e.pos = (int32_t) pos;
            last_rows_[t] = slots[sg];
            last_tokens_[t] = tokens[t];
            last_pos_b_[t] = pos;
        }
        last_seg_slot_[sg] = slots[sg];
        last_seg_len_[sg] = n;
        r0 += n;
    }
    if (do_ple && !ss_->ple.table->gather_batch(ple_rows, (size_t) R, h_ple_, err)) return false;
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = R;
    last_rope_context_ = basalt::kernels::rope_context_index();
    last_nseg_ = S;
    row_base_ = hbase;   // the graphs' key (and nothing else) reads it until the next stage_batch
    last_batch_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                             void* user, int32_t* out, std::string& err) {
    return run_slot_segs(rows, nullptr, S, tokens, pos, pool, user, out, err);
}

// The staged share's RAM addresses (a mapped, 16-byte aligned copy or 0), read again when the source's places moved
// since: between windows, on the window's stream ahead of it.  The last window synchronized its stream, so the host
// copy is free to rewrite.
bool Verifier::dm_stage_places(std::string& err) {
    DecMoe* d = dm_;
    if (d == nullptr || d->stage_src == nullptr) return true;
    const uint64_t gen = warm_src_->places_generation();
    if (gen == d->stage_gen) return true;
    const ModelGeometry& g = *g_;
    const int64_t LE = g.n_layers * g.n_expert;
    bool changed = false;
    for (int64_t i = 0; i < LE; ++i) {
        const unsigned long long p = (unsigned long long) warm_src_->device_alias(i / g.n_expert, i % g.n_expert);
        const unsigned long long v = (p & 15ull) == 0 ? p : 0ull;
        changed = changed || d->stage_host[i] != v;
        d->stage_host[i] = v;
    }
    d->stage_gen = gen;
    if (changed && cudaMemcpyAsync(d->stage_src, d->stage_host, (size_t) LE * 8, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = std::string("verify: the staged share's addresses: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool Verifier::batch_pcie_tunable() const {
    return dm_ != nullptr && dm_->stage_tune;
}

bool Verifier::prepare_batch_pcie(int rows, int pinned_q8, double bandwidth_gbps, double hard_cap, std::string& err) {
    batch_pcie_active_ = batch_pcie_timing_ = false;
    if (!batch_pcie_tunable() || rows <= 0 || hard_cap <= 0.0) return true;
    using namespace basalt::kernels;
    auto& d = *dm_;
    const bool pinned = pinned_q8 >= 0;
    batch_pcie_timing_ = !pinned && d.h_stage_timing != nullptr && bandwidth_gbps > 0.0;
    if (!batch_pcie_timing_ && (!pinned || pinned_q8 == 0)) return true;
    if (batch_pcie_timing_) {
        const uint64_t placement = warm_src_->places_generation();
        if (d.stage_rows != 0 && placement != d.stage_last_placement) ++d.stage_placement_changes;
        d.stage_last_placement = placement;
        // Shape changes change the service estimates. Placement only changes the live
        // miss count and aliases, which the route and dm_stage_places refresh separately.
        if (d.stage_rows != rows) {
            std::fill(d.stage_cost.begin(), d.stage_cost.end(), PcieLayerCost{});
            std::fill(d.stage_misses.begin(), d.stage_misses.end(), 0);
            d.stage_rows = rows;
            ++d.stage_shape_resets;
        }
        d.stage_bandwidth = bandwidth_gbps * 1000.0; // GB/s -> bytes/us
        std::fill_n(d.h_stage_timing, d.stage_cost.size() * DmPcieTimingWords, 0ull);
    }

    const auto& layout = basalt::kernels::cpu::expert_layout();
    const int max_misses = std::min(kDmCap, rows * (int) routed_k_);
    std::fill(d.stage_next.begin(), d.stage_next.end(), 0);
    for (int64_t l = lb_; l < le_; ++l) {
        auto* row = d.stage_next.data() + l * (kDmCap + 1);
        const uint64_t blob = layout.blob_bytes(l);
        const int last_misses = std::min(max_misses, d.stage_misses[l]);
        for (int m = 1; m <= max_misses; ++m) {
            if (pinned) {
                row[m] = (int) std::min<uint64_t>(std::min(d.stage_cap, (int) (m * hard_cap)),
                    m * std::clamp(pinned_q8, 0, 256) * d.stage_reference_bytes / (256 * blob));
            } else {
                PcieBudget reason = PcieBudget::no_timing;
                row[m] = pcie_layer_experts(d.stage_cost[l], m, d.stage_cap,
                                            (double) dec_moe_stage_stride(blob), hard_cap, m == last_misses ? &reason : nullptr);
                if (m == last_misses) ++d.stage_decisions[(size_t) reason];
            }
            row[0] |= row[m] > 0;
        }
        // The previous route decides whether to replay the graph with a staging branch. Its
        // per-layer policy still uses the current window's miss count as the table index.
        batch_pcie_active_ |= pinned ? row[0] != 0 : last_misses > 0 && row[last_misses] > 0;
    }
    if (!batch_pcie_active_ || d.stage_next == d.stage_table) return true;
    const OnDevice on_device(device_);
    const auto status = cudaMemcpyAsync(d.stage_k, d.stage_next.data(), d.stage_next.size() * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, cs_);
    if (status != cudaSuccess) {
        err = std::string("verify: PCIe staging table upload: ") + cudaGetErrorString(status);
        return false;
    }
    d.stage_table.swap(d.stage_next);
    return true;
}

std::string Verifier::take_batch_pcie_budget_report() {
    if (!batch_pcie_tunable()) return {};
    auto& d = *dm_;
    const auto& w = d.stage_decisions;
    if (std::all_of(w.begin(), w.end(), [](uint64_t count) { return count == 0; })) return {};
    char result[256];
    std::snprintf(result, sizeof result, "last-route layer decisions: no_timing=%llu peer_unobserved=%llu "
        "cpu_not_last=%llu hard_cap=%llu no_savings=%llu stage=%llu shape_resets=%llu placement_changes=%llu",
        (unsigned long long) w[(size_t) PcieBudget::no_timing], (unsigned long long) w[(size_t) PcieBudget::peer_unobserved],
        (unsigned long long) w[(size_t) PcieBudget::cpu_not_last], (unsigned long long) w[(size_t) PcieBudget::hard_cap],
        (unsigned long long) w[(size_t) PcieBudget::no_savings], (unsigned long long) w[(size_t) PcieBudget::stage],
        (unsigned long long) d.stage_shape_resets, (unsigned long long) d.stage_placement_changes);
    d.stage_decisions.fill(0);
    d.stage_shape_resets = d.stage_placement_changes = 0;
    return result;
}

Verifier::PcieWindow Verifier::batch_pcie_window() {
    using namespace basalt::kernels;
    PcieWindow result;
    if (!batch_pcie_tunable() || !batch_dm_ || ar_on()) return result;
    auto& d = *dm_;
    const auto& layout = basalt::kernels::cpu::expert_layout();
    std::fill(d.stage_seen.begin(), d.stage_seen.end(), 0);
    for (int64_t l = lb_; l < le_; ++l) {
        const auto& ring = d.h_ring[l - lb_];
        const int entries = ring.n_tok * (int) layer_k(l);
        const int marker = (int) (l - lb_ + 1);
        const uint64_t blob = layout.blob_bytes(l);
        int cpu = 0, local = 0, staged = 0;
        for (int t = 0; t < entries; ++t) {
            const int id = ring.ids[t], kind = ring.kind[t];
            if (id < 0 || id >= g_->n_expert) continue;
            if (kind == 1) ++result.staged_rows;
            if (d.stage_seen[id] == marker) continue;
            d.stage_seen[id] = marker;
            local += kind == 0;
            cpu += kind < 0;
            staged += kind == 1;
            if (kind >= 0 && kind != 1) continue;
            const auto alias = batch_pcie_active_ ? d.stage_host[(size_t) l * g_->n_expert + id]
                : (uintptr_t) warm_src_->device_alias(l, id);
            result.eligible += alias != 0 && (alias & 15u) == 0;
        }
        result.staged += staged;
        result.staged_bytes += staged * blob;
        result.misses += cpu + staged;
        if (!batch_pcie_timing_) continue;
        d.stage_misses[l] = cpu + staged;

        const auto* t = d.h_stage_timing + l * DmPcieTimingWords;
        if (cpu == 0 || t[DmPcieRoute] == 0 || t[DmPcieLocal] == 0) continue;
        auto elapsed_us = [](unsigned long long a, unsigned long long b) {
            return a > b ? (a - b) / 1000.0 : 0.0;
        };
        auto update = [](double& value, double current) {
            value = value > 0.0 ? value * 0.75 + current * 0.25 : current;
        };
        auto& cost = d.stage_cost[l];
        const double per_cpu = d.stage_cpu_us[l] / cpu;
        if (cost.cpu_per_miss > 0.0) update(cost.error_us, std::abs(per_cpu - cost.cpu_per_miss));
        update(cost.cpu_per_miss, per_cpu);
        update(cost.local, elapsed_us(t[DmPcieLocal], t[DmPcieRoute]));
        // The peer acknowledgement is observed by the main combine during its normal CPU wait.
        // A peer that has not acknowledged by then is a conservative floor, not spare slack.
        cost.peer_censored = ring.n_peer > 0 && t[DmPciePeer] == 0;
        if (!cost.peer_censored) update(cost.peer, elapsed_us(t[DmPciePeer], t[DmPcieRoute]));
        if (!cost.measured_stage) {
            cost.bytes_per_us = d.stage_bandwidth * 0.8;
            cost.gpu_per_expert = std::max(60.0, cost.local / std::max(1, local));
        }
        if (staged > 0) {
            const auto copy_first = *std::min_element(t + DmPcieCopyBegin, t + DmPcieCopyEnd);
            const auto copy_last = *std::max_element(t + DmPcieCopyEnd, t + DmPcieTimingWords);
            const auto copy_us = elapsed_us(copy_last, copy_first);
            const auto compute_us = elapsed_us(t[DmPcieGrouped], std::max(t[DmPcieLocal], copy_last));
            if (copy_first != 0 && copy_us > 0.0 && compute_us > 0.0) {
                update(cost.copy_start, elapsed_us(copy_first, t[DmPcieRoute]));
                update(cost.bytes_per_us, std::min(d.stage_bandwidth, staged * (double) blob / copy_us));
                update(cost.gpu_per_expert, compute_us / staged);
                cost.measured_stage = true;
            }
        }
    }
    return result;
}

bool Verifier::run_slot_segs(const int* rows, const int* lens, int S, const int32_t* tokens, const int64_t* pos,
                             PoolMultiFn pool, void* user, int32_t* out, std::string& err) {
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (gs_live_ && !gs_move(false, err)) return false;
    gs_split_ = false;
    if (!stage_batch(rows, lens, S, 0, tokens, pos, err)) return false;
    const int R = last_t_;   // the window's rows
    const bool dm = dm_ != nullptr && batch_dm_;
    if (dm && !dm_begin(err, nullptr)) return false;
    if (dm && (!dm_->stage_tune || batch_pcie_active_) && !dm_stage_places(err)) return false;
    order_after_commit();
    set_head_keep(0);   // the slots sample with their own parameters: every row exact
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    const cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(R, 0)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = ar_on() ? 0 : le_ - lb_;   // the zero-doorbell graph rings no layer
    if (ar_on()) ar_handoff();
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        progress_at("verify batch: the CPU experts of layer", l);
        if (dm) {   // the ring's own rows and tiers: the pool computes the CPU's entries only
            const basalt::kernels::DmRing& Rg = dm_->h_ring[k];
            sink_.ext_kind = Rg.kind;
            if (pool != nullptr) pool(user, dm_->h_x + (size_t) k * dm_->x_stride, Rg.ids, R, routed_k_, h_ymiss_, l);
            sink_.ext_kind = nullptr;
        } else if (pool != nullptr) pool(user, h_x_, h_ids_, R, routed_k_, h_ymiss_, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        const double pool_ms = ms_since(b);
        ms_pool += pool_ms;
        if (dm && batch_pcie_timing_) dm_->stage_cpu_us[l] = pool_ms * 1000.0;
    }
    progress_at("verify batch: waiting for the GPU to finish the window", R);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    const bool dm_ok = !dm || dm_end(err);
    if (!window_intact(err) || !dm_ok) return false;
    if (se != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(se); return false; }
    if (!ar_plan_intact(err)) return false;
    report_dm_wait(R);
    if (prof_on_) collect_profile();
    ++windows;
    if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_segs(rows, lens, S, tokens, pos, pool, next_user_, out, err);
    if (out != nullptr) {
        if (!sample_rows(R, err)) return false;
        for (int t = 0; t < R; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    }
    progress_at("decode");
    progress_beat();
    return true;
}

bool Verifier::commit_slots(std::string& err) { return commit_slots(nullptr, err); }

bool Verifier::commit_slots(const int* n_keep, std::string& err) {
    return commit_slots_begin(n_keep, err) && commit_slots_end(err);
}

bool Verifier::commit_slots_begin(const int* n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    commit_t0_ = Clock::now();
    const int64_t CB = 2 + max_t_;
    if (n_keep != nullptr) {
        for (int sg = 0; sg < last_nseg_; ++sg)
            if (n_keep[sg] < 1 || n_keep[sg] > last_seg_len_[sg]) { err = "verify: commit count out of range"; return false; }
        for (int sg = 0, r0 = 0; sg < last_nseg_; r0 += last_seg_len_[sg], ++sg) {
            const int k = n_keep[sg], n = last_seg_len_[sg];
            for (int j = 0; j < n; ++j) h_segb_[r0 + j].n_keep = k;
            int32_t* c = h_commitb_ + (size_t) sg * CB;
            c[0] = k;
            c[1] = k - 1;
            for (int j = k; j < n; ++j) c[2 + j] = -1;
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    order_after_commit();
    const cudaError_t le = cudaGraphLaunch(commit_bm_[commit_batch_key(last_t_, last_rope_context_)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch commit launch: ") + cudaGetErrorString(le); return false; }
    // (the PLE history the commit leaves is known from the tokens: it advances now, the graph runs beside the caller)
    if (ple_stage())
        for (int sg = 0, r0 = 0; sg < last_nseg_; r0 += last_seg_len_[sg], ++sg) {
            SessionState& sx = *slots_[(size_t) last_seg_slot_[sg]];
            const int k = n_keep != nullptr ? n_keep[sg] : last_seg_len_[sg];
            for (int j = 0; j < k; ++j) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[r0 + j];
            }
        }
    commit_next_keep_.assign(n_keep != nullptr ? n_keep : nullptr, n_keep != nullptr ? n_keep + last_nseg_ : nullptr);
    return true;
}

bool Verifier::commit_slots_end(std::string& err) {
    const OnDevice on_device(device_);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }
    ms_commit += ms_since(commit_t0_);
    return next_ == nullptr || next_->commit_slots(commit_next_keep_.empty() ? nullptr : commit_next_keep_.data(), err);
}

bool Verifier::sample_rows(int S, std::string& err) {
    bool any = false, wide = false;
    for (int t = 0; t < S; ++t) {
        const basalt::kernels::SamplerParams& sp = slot_sp_[(size_t) last_rows_[t]];
        wide = wide || (!sp.greedy && sp.temperature > 0.0f &&
                        basalt::kernels::sampler_list_len(sp.top_k, (int) n_vocab_) > basalt::kernels::kSamplerNarrowMax);
    }
    if (wide && !exact_head_rows(S, err)) return false;
    for (int t = 0; t < S; ++t) {
        basalt::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        const bool pen = bh_d_ != nullptr && sp.penalty_last_n > 0;
        if ((sp.greedy || sp.temperature <= 0.0f) && !pen) continue;   // the recorded greedy pick stands
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        if (!pen) sp.penalty_last_n = 0;
        basalt::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_,
                                       pen ? bh_d_ + (size_t) t * (size_t) bh_len_ : nullptr, pen ? bh_len_ : 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (gs_live_ && !gs_move(false, err)) return false;
    gs_split_ = false;
    if (!stage_batch(rows, nullptr, S, base, tokens, pos, err)) return false;
    order_after_commit();
    set_head_keep(0);   // the slots sample with their own parameters: every row exact
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(S, base)], cs_);
    if (le == cudaSuccess) le = cudaGraphLaunch(commit_bm_[commit_batch_key(S, basalt::kernels::rope_context_index())], cs_);   // right behind it: every row is kept
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    if (ar_on()) ar_handoff();
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = ar_on() ? 0 : le_ - lb_;   // the zero-doorbell graph rings no layer
    b_last_ = Clock::now();
    return true;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    volatile uint32_t* const seq = h_seq_;
    const int S = last_t_;
    while (b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (*seq < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > std::chrono::seconds(20)) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) + released_note(release_gpu_waits(5000));
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, routed_k_, h_ymiss_, lb_ + b_k_);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    const cudaError_t q = cudaStreamQuery(cs_);
    if (q == cudaErrorNotReady) {
        if (Clock::now() - b_last_ > std::chrono::seconds(20)) {
            err = "verify batch: timed out at the window's end" + released_note(release_gpu_waits(5000));
            b_running_ = false;
            return -1;
        }
        return 0;
    }
    if (q != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(q); b_running_ = false; return -1; }
    if (!window_intact(err) || !ar_plan_intact(err)) { b_running_ = false; return -1; }
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}

}  // namespace basalt::core
