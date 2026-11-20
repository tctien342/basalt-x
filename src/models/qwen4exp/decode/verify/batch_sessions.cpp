#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/models/qwen4exp/kernels/dec_gdn.hpp"
#include "basalt/models/qwen4exp/kernels/native_ple_postops.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_dec_fused.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

namespace {
bool batch_kv_format_matches(const QsaState& first, const QsaState& st) {
    return st.kv_mode <= 1 && st.kv_int8 == first.kv_int8 && st.kv_rot == first.kv_rot &&
           (st.kv_int8 || !st.kv_rot) && !st.kv_hybrid && !st.kv_q4;
}

bool shared_kv_backing_matches(const QsaState& first, const QsaState& st) {
    const auto& a = first.map;
    const auto& b = st.map;
    if (!st.shared_residency || st.kv_mode != 1 || !st.host.fp16_host_pages ||
        !b.page_table || !b.ctl || st.page_table != b.page_table ||
        b.page_table != a.page_table || b.ctl != a.ctl || b.slot_block != a.slot_block ||
        b.slot_stamp != a.slot_stamp || b.slot_ref != a.slot_ref || b.miss_block != a.miss_block ||
        b.miss_slot != a.miss_slot || b.n_slots != st.n_slots || b.n_slots != a.n_slots ||
        b.n_blocks <= 0 || b.n_blocks != a.n_blocks || size_t(b.n_blocks) != st.host_pool_pages)
        return false;

    if (st.kv_int8)
        return st.k_q && st.v_q && st.k_scale && st.v_scale &&
               st.k_q == first.k_q && st.v_q == first.v_q && st.k_scale == first.k_scale &&
               st.v_scale == first.v_scale && st.host.k_q && st.host.v_q && st.host.k_scale &&
               st.host.v_scale && st.host.k_q == first.host.k_q && st.host.v_q == first.host.v_q &&
               st.host.k_scale == first.host.k_scale && st.host.v_scale == first.host.v_scale;
    return st.k_pool && st.v_pool && st.k_pool == first.k_pool && st.v_pool == first.v_pool &&
           st.host.k_pool && st.host.v_pool && st.host.k_pool == first.host.k_pool &&
           st.host.v_pool == first.host.v_pool;
}
}  // namespace


// ================================ BATCH WINDOWS (see init_slots) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    const bool shared = ss_->qsa_alloc > 0 && ss_->qsa_states[ss_->qsa_primary()].shared_residency;
    if (slots.empty() || (!shared && (int) slots.size() > max_t_)) {
        err = "verify: invalid session count for the selected residency mode";
        return false;
    }
    for (SessionState* x : slots) {
        if (x == nullptr || x->layer_lo != ss_->layer_lo || x->layer_hi != ss_->layer_hi ||
            x->max_cells > ss_->max_cells || x->gdn_ord0 != ss_->gdn_ord0 || x->qsa_ord0 != ss_->qsa_ord0) {
            err = "verify: a slot's session is not carved like the stage's own (layer range, a context at most the stage's)";
            return false;
        }
    }
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    void* d = nullptr;
    if (!mapped((size_t) (max_t_ * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    if (!mapped((size_t) max_t_ * sizeof(basalt::kernels::SegRow), (void**) &h_segb_, (void**) &m_segb_)) {
        err = "verify: the batch segment staging failed";
        return false;
    }
    if (!mapped((size_t) max_t_ * sizeof(float*), (void**) &h_ple_slots_, (void**) &m_ple_slots_)) {
        err = "verify: the batch PLE history staging failed";
        return false;
    }
    using namespace basalt::kernels;
    if (!mapped((size_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg), (void**) &h_qseg_, (void**) &m_qseg_)) {
        err = "verify: the batch QSA table staging failed";
        return false;
    }
    std::fill_n(h_qseg_, basalt::kernels::kQsaSegMaxRows + 8, QsaSeg{});
    // Every row in a segmented launch uses the same storage format and rotation.
    if (!(qsa_dec_fused_supported(s))) {
        err = "verify: batch windows need the native QSA kernels";
        return false;
    }
    batch_stream_ = false;
    bool checked_layer = false;
    batch_shared_residency_ = false;
    for (int64_t l = lb_; l < le_; ++l) {
        if (!is_qsa_layer(*g_, l)) continue;
        int64_t qi = 0;
        for (int64_t m = 0; m < l; ++m) qi += is_qsa_layer(*g_, m) ? 1 : 0;
        const QsaState& first = slots.front()->qsa_states[qi];
        if (!checked_layer) batch_shared_residency_ = first.shared_residency;
        checked_layer = true;
        for (SessionState* x : slots) {
            const QsaState& st = x->qsa_states[qi];
            if (st.shared_residency != batch_shared_residency_) {
                err = "verify: batch slots and layers must agree on shared residency";
                return false;
            }
            if (!batch_kv_format_matches(first, st)) {
                err = "verify: batch windows need a common FP16 or INT8 K/V format and rotation";
                return false;
            }
            if (batch_shared_residency_ && !shared_kv_backing_matches(first, st)) {
                err = "verify: shared slots need the same physical residency map and backing";
                return false;
            }
            if (st.kv_mode == 1 && st.n_slots < basalt::kernels::kKvStreamMinSlots) {   // one step of the resolve's sweep
                err = "verify: a streamed slot holds too few pages for the resolve; raise --kv-resident";
                return false;
            }
            batch_stream_ = batch_stream_ || st.kv_mode == 1;
        }
    }
    const size_t descriptors = (size_t) std::max<int64_t>(nQ, 1) * max_t_;
    if (!mapped(descriptors * sizeof(QsaSegSlot), (void**) &h_qslots_, (void**) &m_qslots_)) {
        err = "verify: the batch QSA descriptor staging failed";
        return false;
    }
    const uint64_t a = ((uint64_t) max_t_ * CB * 4 + 255) & ~255ull;
    const uint64_t b = (a + (uint64_t) descriptors * TS * 4 + 255) & ~255ull;
    const uint64_t b2 = (b + (uint64_t) max_t_ * sizeof(basalt::kernels::SegRow) + 255) & ~255ull;
    const uint64_t b3 = (b2 + (uint64_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg) + 255) & ~255ull;
    const uint64_t HSb = (uint64_t) NG_HIST * NG_HC_DIM * sizeof(float);
    const uint64_t b4 = (b3 + (uint64_t) max_t_ * sizeof(float*) + 255) & ~255ull;
    const uint64_t b5 = (b4 + (uint64_t) max_t_ * HSb + 255) & ~255ull;
    if (cudaMalloc(&d, b5 + descriptors * sizeof(QsaSegSlot)) != cudaSuccess) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    arena_b_ = d;
    commitb_ = (int32_t*) d;
    tail_snap_b_ = (float*) ((uint8_t*) d + a);
    segb_ = (basalt::kernels::SegRow*) ((uint8_t*) d + b);
    d_qrows_ = (QsaSeg*) ((uint8_t*) d + b2);
    d_qsegs_ = d_qrows_ + basalt::kernels::kQsaSegMaxRows;
    d_ple_slots_ = (float**) ((uint8_t*) d + b3);
    ple_work_ = (float*) ((uint8_t*) d + b4);
    d_qslot_base_ = (QsaSegSlot*) ((uint8_t*) d + b5);
    d_qslots_.resize((size_t) nQ);
    for (int64_t qi = 0; qi < nQ; ++qi)
        d_qslots_[(size_t) qi] = d_qslot_base_ + (size_t) qi * max_t_;
    slots_ = slots;
    batch_dm_ = dm_ != nullptr;
    slot_adapter_scale_.assign(slots.size(), 0.0f);
    slot_swift_scale_.assign(slots.size(), 0.0f);
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    basalt::log::info("basalt verify", "batch windows of up to %lld sequences (layers [%lld, %lld))%s", (long long) S,
                 (long long) lb_, (long long) le_, batch_stream_ ? ", the slots' K/V streamed from RAM" : "");
    if (dm_ != nullptr && dm_->stage_tune)
        basalt::log::info("basalt verify", "batch windows: adaptive PCIe staging with one peer");
    else if (dm_ != nullptr && dm_->stage_src != nullptr)
        basalt::log::info("basalt verify", "batch windows: a layer with %d+ CPU misses sends a share of them over PCIe to "
                          "CUDA0", dm_->stage_min);
    return true;
}

bool Verifier::append_shared_slot(SessionState& slot, std::string& err) {
    if (!batch_shared_residency_ || slots_.empty()) {
        err = "verify: shared slots must be initialized before appending";
        return false;
    }
    const SessionState& owner = *slots_.front();
    if (slot.layer_lo != owner.layer_lo || slot.layer_hi != owner.layer_hi ||
        slot.max_cells > owner.max_cells || slot.gdn_ord0 != owner.gdn_ord0 ||
        slot.qsa_ord0 != owner.qsa_ord0 || slot.qsa_alloc != owner.qsa_alloc) {
        err = "verify: shared slot geometry differs from its owner";
        return false;
    }
    for (int64_t q = owner.qsa_ord0; q < owner.qsa_ord0 + owner.qsa_alloc; ++q) {
        const auto& a = owner.qsa_states[q];
        const auto& b = slot.qsa_states[q];
        if (!batch_kv_format_matches(a, b) || !shared_kv_backing_matches(a, b)) {
            err = "verify: appended slot does not match the shared K/V format and backing";
            return false;
        }
    }
    const auto count = slots_.size() + 1;
    slots_.reserve(count);
    slot_sp_.reserve(count);
    slot_adapter_scale_.reserve(count);
    slot_swift_scale_.reserve(count);
    slots_.push_back(&slot);
    slot_sp_.push_back(sampling_);
    slot_adapter_scale_.push_back(0.0f);
    slot_swift_scale_.push_back(0.0f);
    return true;
}

void Verifier::undo_shared_slot_append() {
    slots_.pop_back();
    slot_sp_.pop_back();
    slot_adapter_scale_.pop_back();
    slot_swift_scale_.pop_back();
}

bool Verifier::bind_session(SessionState* s, std::string& err) {
    if (s == nullptr || ss_ == nullptr) { err = "verify: bind_session without a session"; return false; }
    if (s == ss_) return true;
    if (!wait_commit(err)) return false;   // the last commit still works on the old session's state
    SoloGraphs cur;
    std::copy(std::begin(exec_), std::end(exec_), std::begin(cur.exec));
    std::copy(std::begin(exec_nr_), std::end(exec_nr_), std::begin(cur.exec_nr));
    std::copy(std::begin(commit_exec_), std::end(commit_exec_), std::begin(cur.commit));
    solo_sets_[ss_] = cur;
    SoloGraphs nx;
    const auto it = solo_sets_.find(s);
    if (it != solo_sets_.end()) {
        nx = it->second;
        solo_sets_.erase(it);
    }
    std::copy(std::begin(nx.exec), std::end(nx.exec), std::begin(exec_));
    std::copy(std::begin(nx.exec_nr), std::end(nx.exec_nr), std::begin(exec_nr_));
    std::copy(std::begin(nx.commit), std::end(nx.commit), std::begin(commit_exec_));
    ss_ = s;
    return true;
}

bool Verifier::warm_session(SessionState* s, std::string& err) {
    SessionState* home = ss_;
    if (!bind_session(s, err)) return false;
    refresh_ar();
    bool ok = capture_commit(err);
    for (int T = 1; ok && T <= max_t_; ++T) ok = capture(T, err);
    std::string e2;
    if (!bind_session(home, e2) && ok) { err = e2; return false; }
    return ok;
}

bool Verifier::warm_graphs(std::string& err) {
    refresh_ar();
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (slots_.empty()) return true;
    for (int R = 1; R <= max_t_; ++R)
        if (!capture_batch(R, 0, err) || !capture_commit_batch(R, err)) return false;
    return true;
}

}  // namespace basalt::core
