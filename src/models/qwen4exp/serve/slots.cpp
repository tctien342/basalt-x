// src/models/qwen4exp/serve/slots.cpp - slots operations.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"

namespace basalt::models::qwen4exp::serve {
namespace {

bool resize_shared_index(const basalt::core::SharedKvTable::Stage& stage, int64_t cells,
                         bool grow, std::string& err) {
    const basalt::core::OnDevice device(stage.device);
    for (int64_t q = 0; q < stage.session.qsa_alloc; ++q) {
        auto* arena = stage.session.qsa_states[stage.session.qsa_ord0 + q].shared_index_arena;
        if (arena && !(grow ? arena->map(cells, err) : arena->unmap(cells, err))) return false;
    }
    return true;
}

}  // namespace

// a view of the solo session carrying slot b's state: what the request path and the solo verifier work on for a
// request that runs in that slot
bool QfnRuntime::settle_shared_slot(int slot, bool retain, int64_t solo_cells) {
    if (!shared_pages) return true;
    const bool retire_frozen = slot == 0 && frozen_recovery.valid();
    if (retire_frozen) finish_frozen_recovery();
    auto& sequence = *shared_sequences[(size_t) slot];
    if (sequence.lease.pages().empty()) return true;
    auto& state = bs[(size_t) slot];
    if (!drain_windows() || !ver.wait_commit(err) ||
        (thinking_ver && !thinking_ver->wait_commit(err))) return false;
    // Lease backing can also be written by prompt and draft streams.
    if (const auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
        err = cudaGetErrorString(status);
        return false;
    }
    const auto page = basalt::kernels::qsa_real_shapes().page_size;
    const auto cells = solo_cells >= 0 ? (size_t) solo_cells : state.cached ? state.ids.size() : 0;
    const auto keep = retain ? (cells + page - 1) / page : 0;
    using Table = basalt::core::SharedKvTable;
    const auto status = keep ? Table::park_prefix(*shared_pages, sequence.lease, keep, sequence.stages)
                             : Table::retire(*shared_pages, sequence.lease, sequence.stages);
    if (status != cudaSuccess) { err = cudaGetErrorString(status); return false; }
    std::vector<int32_t>().swap(sequence.image_rope);
    for (const auto& stage : sequence.stages)
        if (!resize_shared_index(stage, keep * page, false, err)) return false;
    if (!keep) {
        state.cached = false;
        state.ids.clear();
        state.imgs.clear();
        state.checks.clear();
        // Slot zero's solo checkpoint chain is separate from its batch metadata.
        // Retiring the page lease invalidates both, even when a peer owns the live request.
        if (slot == 0) {
            (solo_out ? solo_out->live : live).clear();
            (solo_out ? solo_out->live_imgs : live_imgs).clear();
            (solo_out ? solo_out->checks : checks).clear();
            (solo_out ? solo_out->live_ok : live_ok) = false;
        }
        if (batch_spec && !mtp.retire_shared_store(slot, err)) return false;
    }
    if (basalt::core::exp_flag("shared_kv_audit_parked")) {
        for (size_t i = 0; i < shared_sequences.size(); ++i) {
            if ((i != (size_t) slot && bs[i].active) || !shared_sequences[i]) continue;
            const auto ids = shared_sequences[i]->lease.pages();
            const size_t used = std::min(i == (size_t) slot ? cells : bs[i].ids.size(), ids.size() * page);
            if (used < (size_t) page) continue;
            uint64_t pages_hash = 1469598103934665603ull;
            for (const auto id : ids) pages_hash = (pages_hash ^ (uint64_t) id) * 1099511628211ull;
            uint64_t part[10] = {};
            if (!slot_digest((int) i, (int64_t) used, part)) {
                basalt::log::warn("basalt KV audit", "finished=%d slot=%zu digest failed", slot, i);
                continue;
            }
            char parts[192] = {};
            size_t pos = 0;
            for (uint64_t x : part)
                pos += (size_t) std::snprintf(parts + pos, sizeof(parts) - pos, " %016llx", (unsigned long long) x);
            basalt::log::info("basalt KV audit", "finished=%d slot=%zu cells=%zu pages=%016llx state=%s",
                              slot, i, used, (unsigned long long) pages_hash, parts);
        }
    }
    if (retire_frozen) {
        std::printf("INFO frozen_shared_kv_pages=0 frozen_shared_kv_cache=0\n");
        std::fflush(stdout);
    }
    return true;
}

Flow QfnRuntime::publish_shared_request(int slot, uint32_t pages, unsigned adapter) {
    if (!shared_pages) return Flow::proceed;
    if (slot < 0 || (size_t) slot >= shared_sequences.size() || !shared_sequences[(size_t) slot]) {
        std::printf("ERR shared KV: missing sequence binding\n");
        return Flow::error;
    }
    if (slot == 0 && frozen_recovery.valid() && !settle_shared_slot(0, false)) {
        std::printf("ERR shared KV: frozen source retirement failed: %s\n", err.c_str());
        return Flow::error;
    }
    auto& sequence = *shared_sequences[(size_t) slot];
    auto& lease = sequence.lease;
    // The API only charges this request's extent. Drop an old prefix that cannot fit within that charge.
    const auto context = basalt::kernels::rope_context();
    if (!lease.pages().empty() && (sequence.adapter != adapter || sequence.rope_context != context ||
                                  lease.pages().size() > pages)) {
        if (!settle_shared_slot(slot, false)) {
            std::printf("ERR shared KV: prefix retirement failed: %s\n", err.c_str());
            return Flow::error;
        }
    }
    sequence.adapter = adapter;
    sequence.rope_context = context;
    // A resumed request keeps its page IDs. Its existing prefix can extend
    // only after the pool has reserved the entire new extent.
    const size_t first_victim = frozen_recovery.valid() ? 1 : 0;
    for (size_t b = first_victim; b < shared_sequences.size() &&
         pages - lease.pages().size() > shared_pages->available(); ++b) {
        if (b == (size_t) slot || bs[b].active) continue;
        if (!settle_shared_slot((int) b, false)) {
            std::printf("ERR shared KV: idle eviction failed: %s\n", err.c_str());
            return Flow::error;
        }
    }
    if (pages - lease.pages().size() > shared_pages->available() && frozen_recovery.valid() &&
        !settle_shared_slot(0, false)) {
        std::printf("ERR shared KV: frozen source retirement failed: %s\n", err.c_str());
        return Flow::error;
    }
    const bool reserved = lease.pages().empty() ? shared_pages->reserve(lease, pages)
                                               : shared_pages->resize_reservation(lease, pages);
    if (!reserved) {
        std::printf("ERR shared KV: insufficient free pages\n");
        std::fflush(stdout);
        return Flow::next;
    }
    const int64_t page_cells = basalt::kernels::qsa_real_shapes().page_size;
    uint64_t index_bytes = 0;
    for (const auto& stage : sequence.stages)
        for (int64_t q = 0; q < stage.session.qsa_alloc; ++q)
            if (const auto* arena = stage.session.qsa_states[stage.session.qsa_ord0 + q].shared_index_arena)
                index_bytes += arena->grow_bytes(pages * page_cells);
    const uint64_t draft_bytes = batch_spec ? mtp.shared_store_bytes(slot) : 0;
    const uint64_t backing_bytes = index_bytes + draft_bytes;
    // Leave the configured reserve for CUDA's lazy prefill kernels and graph setup.
    const uint64_t reserve = (uint64_t) std::max<int64_t>(0, o.vram_reserve_mib) << 20;
    if (backing_bytes && !shared_expert_room(backing_bytes + reserve, err)) {
        std::printf("ERR shared KV: expert lending failed: %s\n", err.c_str());
        return Flow::error;
    }
    size_t victim = frozen_recovery.valid() ? 1 : 0;
    for (;;) {
        const bool draft_backed = !batch_spec || mtp.prepare_shared_stores(slot, err);
        bool backed = draft_backed;
        for (const auto& stage : sequence.stages) {
            if (!backed) break;
            if (!resize_shared_index(stage, pages * page_cells, true, err)) {
                backed = false;
                break;
            }
        }
        if (backed) break;
        const bool no_vram = !draft_backed || err == "the driver has no VRAM for the KV pools";
        if (no_vram) {
            while (victim < bs.size() && (victim == (size_t) slot || bs[victim].active ||
                   shared_sequences[victim]->lease.pages().empty())) ++victim;
            if (victim < bs.size()) {
                if (settle_shared_slot((int) victim++, false)) continue;
                std::printf("ERR shared KV: idle eviction failed: %s\n", err.c_str());
                return Flow::error;
            }
            if (frozen_recovery.valid()) {
                if (settle_shared_slot(0, false)) continue;
                std::printf("ERR shared KV: frozen source retirement failed: %s\n", err.c_str());
                return Flow::error;
            }
        }
        const auto prior = lease.pages().size();
        for (const auto& stage : sequence.stages)
            if (!resize_shared_index(stage, prior * page_cells, false, err)) {
                std::printf("ERR shared KV: indexer rollback failed: %s\n", err.c_str());
                return Flow::error;
            }
        if (prior) shared_pages->resize_reservation(lease, prior);
        else shared_pages->release(lease);
        std::printf("ERR shared KV: %s backing unavailable: %s\n",
                    draft_backed ? "indexer" : "draft", err.c_str());
        std::fflush(stdout);
        return no_vram ? Flow::next : Flow::error;
    }
    const auto status = basalt::core::SharedKvTable::publish_extent(*shared_pages, lease, pages, sequence.stages);
    if (status == cudaSuccess) {
        if (frozen_recovery.valid())
            basalt::log::info("basalt serve", "cold shared snapshot: admitted slot %d while holding source slot zero", slot);
        return Flow::proceed;
    }
    std::printf("ERR shared KV: page publication failed: %s\n", cudaGetErrorString(status));
    return Flow::error;
}

bool QfnRuntime::append_shared_session(std::string& error) {
    using namespace basalt::core;
    if (!shared_pages || bslot_ss.size() != 1 || piped) {
        error = "shared slot growth requires the unsplit decode scheduler";
        return false;
    }
    if (!ver.wait_commit(error)) return false;
    const OnDevice device(0);
    const auto bytes = session_bytes(g, ss.max_cells, ss.k, ss.layer_lo, ss.layer_hi, true);
    const auto table_bytes = (uint64_t) shared_pages->capacity() * sizeof(int32_t);
    if (!shared_expert_room(bytes + table_bytes + (batch_spec ? mtp.shared_store_bytes() : 0), error))
        return false;
    const auto count = bs.size() + 1;
    bs.reserve(count);
    bscript.reserve(count);
    bslot_ss[0].reserve(count);
    slot_cap.reserve(count);
    shared_tables.reserve(count);
    shared_sequences.reserve(count);
    auto table = std::make_unique<SharedKvTable>();
    if (const auto status = table->allocate(shared_pages->capacity()); status != cudaSuccess) {
        if (status == cudaErrorMemoryAllocation) cudaGetLastError();
        error = cudaGetErrorString(status);
        return false;
    }
    std::vector<QsaHostBinding> bindings;
    for (int64_t q = 0; q < ss.qsa_alloc; ++q)
        bindings.push_back({ss.qsa_states[ss.qsa_ord0 + q], table->device_pages(), nullptr, 0, true});
    auto sequence = std::make_unique<SharedSequence>();
    sequence->stages.reserve(1);
    engine_local::SlotSession slot(new SessionState());
    void*& arena = slot.get_deleter().runtime_arena;
    if (const auto status = cudaMalloc(&arena, bytes); status != cudaSuccess) {
        if (status == cudaErrorMemoryAllocation) cudaGetLastError();
        error = cudaGetErrorString(status);
        return false;
    }
    if (!session_init(g, ss.max_cells, ss.k, arena, *slot, ss.layer_lo, ss.layer_hi, bindings)) {
        error = "shared slot session initialization failed";
        return false;
    }
    session_zero(*slot, g, nullptr, nullptr);
    if (const auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    bind_lora_ple(*slot, lora_main.get());
    sequence->stages.push_back({0, *table, *slot, nullptr});
    if (!ver.append_shared_slot(*slot, error)) return false;
    if (batch_spec) {
        cudaError_t status = cudaSuccess;
        if (!mtp.append_shared_store(ss.max_cells, error, status)) {
            // A forecast cannot account for all driver allocation overhead. Retry only
            // after actually returning an expert segment and only for device OOM.
            const auto before = xcache.mapped_bytes();
            std::string lending_error;
            const bool reclaimed = status == cudaErrorMemoryAllocation &&
                shared_expert_room(mtp.shared_store_bytes(), lending_error, true) &&
                xcache.mapped_bytes() < before;
            if (reclaimed)
                basalt::log::info("basalt serve", "retrying shared slot growth after %s", error.c_str());
            if (!reclaimed || !mtp.append_shared_store(ss.max_cells, error, status)) {
                if (!lending_error.empty()) error = lending_error;
                ver.undo_shared_slot_append();
                return false;
            }
        }
    }
    slot_cap.push_back(ss.max_cells);
    rviews.emplace_back();
    bslot_ss[0].push_back(std::move(slot));
    shared_tables.push_back(std::move(table));
    shared_sequences.push_back(std::move(sequence));
    bs.emplace_back();
    bscript.emplace_back();
    return true;
}

void QfnRuntime::make_view(basalt::core::SessionState& v, int b) {
    const basalt::core::SessionState& t = *bslot_ss[0][(size_t) b];
    v = ss;
    v.gdn_state = t.gdn_state;
    v.qsa_states = t.qsa_states;
    v.ple_hist = t.ple_hist;
    v.ple.hist = t.ple_hist;
    v.ple_prev[0] = t.ple_prev[0];
    v.ple_prev[1] = t.ple_prev[1];
    v.ple_token = t.ple_token;
    v.ple.token = &v.ple_token;
    v.ple.prev = v.ple_prev;
}

// the session a request just left behind (its prompt) -> slot b's sessions, on every stage
bool QfnRuntime::copy_to_slot(int b, const std::vector<int32_t>& ids, std::string& e) {
    if (b == 0 && frozen_recovery.valid()) {
        e = "cold shared snapshot owns slot zero";
        return false;
    }
    const int64_t upto = (int64_t) ids.size();
    for (size_t k = 0; k < bslot_ss.size(); ++k) {
        basalt::core::SessionState& from = k == 0 ? *rsp : stages[k - 1]->ss;
        basalt::core::SessionState& to = *bslot_ss[k][(size_t) b];
        if (from.gdn_state == to.gdn_state) continue;   // the same state (slot 0, or the request's own slot)
        const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
        // the elastic KV: the slot backs the copied cells and its room past them, and gives back the rest
        if (k == 0 && (!kv_fit(to, upto + kKvGrowCells, e) || !kv_release(to, upto + kKvGrowCells, e)))
            return false;
        if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
        basalt::core::ConversationCheckpoint ck;
        ck.ids = ids;
        if (!basalt::core::conversation_checkpoint_save(ck, from, g, e) ||
            !basalt::core::conversation_checkpoint_restore(ck, to, g, e))
            return false;
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            basalt::core::ConversationKv img;
            if (!basalt::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto, true, e))
                return false;
            if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
            if (!basalt::core::conversation_kv_restore(img, to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
    }
    if (batch_spec && !mtp.copy_store(mtp.main_store(), b, upto, e)) return false;   // the draft layer's K/V
    return true;
}

bool QfnRuntime::hold_slot(int b, const ConvCheckpoint* at, std::string& e) {
    slot_held = SlotHeld{};
    const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
    slot_held.ck.resize(bslot_ss.size());
    slot_held.kv.resize(bslot_ss.size());
    for (size_t k = 0; k < bslot_ss.size(); ++k) {
        basalt::core::SessionState& from = *bslot_ss[k][(size_t) b];
        const basalt::core::SessionState& to = k == 0 ? *rsp : stages[k - 1]->ss;
        if (from.gdn_state == to.gdn_state) continue;
        const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
        if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot hold: device sync failed"; return false; }
        if (at == nullptr) {
            slot_held.ck[k].ids = bs[(size_t) b].ids;
            if (!basalt::core::conversation_checkpoint_save(slot_held.ck[k], from, g, e)) return false;
        }
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            slot_held.kv[k].emplace_back();
            if (!basalt::core::conversation_kv_save(slot_held.kv[k].back(), from.qsa_states[from.qsa_ord0 + j], g,
                                                    upto, true, e))
                return false;
            if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot hold: device sync failed"; return false; }
        }
    }
    slot_held.b = b;
    return true;
}

// slot b's sessions -> the main ones (the reverse of copy_to_slot): a request that continues the
// conversation an idle slot holds reads on from there.  Every stage, the same calls.
// `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
bool QfnRuntime::copy_from_slot(int b, const ConvCheckpoint* at, std::string& e) {
    const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
    if (at != nullptr && at->stage_parts.size() != stages.size()) { e = "a checkpoint without its stage parts"; return false; }
    const bool held = slot_held.b == b;
    for (size_t k = 0; k < bslot_ss.size(); ++k) {
        basalt::core::SessionState& from = *bslot_ss[k][(size_t) b];
        basalt::core::SessionState& to = k == 0 ? *rsp : stages[k - 1]->ss;
        if (from.gdn_state == to.gdn_state) continue;   // the same state (slot 0, or the request's own slot)
        const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
        if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
        if (at != nullptr) {
            if (!basalt::core::conversation_checkpoint_restore(k == 0 ? *at : at->stage_parts[k - 1], to, g, e))
                return false;
        } else if (held) {
            if (!basalt::core::conversation_checkpoint_restore(slot_held.ck[k], to, g, e)) return false;
        } else {
            basalt::core::ConversationCheckpoint ck;
            ck.ids = bs[(size_t) b].ids;
            if (!basalt::core::conversation_checkpoint_save(ck, from, g, e) ||
                !basalt::core::conversation_checkpoint_restore(ck, to, g, e))
                return false;
        }
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            // the slot's cells straight into the request's session (a held slot's from its RAM copy)
            if (held ? !basalt::core::conversation_kv_restore(slot_held.kv[k][(size_t) j],
                                                              to.qsa_states[to.qsa_ord0 + j], g, upto, true, e)
                     : !basalt::core::conversation_kv_copy(from.qsa_states[from.qsa_ord0 + j],
                                                           to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
    }
    if (held) slot_held = SlotHeld{};
    if (batch_spec && !mtp.copy_store(b, mtp.main_store(), upto, e)) return false;
    return true;
}

// the solo conversation's draft K/V from slot 0's store back into the main sequence once a request in another slot
// has drafted there; `keep` false: the solo session gave its cells to that request, nothing comes back
void QfnRuntime::solo_draft_back(bool keep) {
    if (!solo_draft_in0) return;
    solo_draft_in0 = false;
    std::string e;
    if (keep && live_ok && !live.empty() && !mtp.copy_store(0, mtp.main_store(), (int64_t) live.size(), e))
        basalt::log::warn("basalt serve", "the solo conversation's draft K/V: %s", e.c_str());
}

// hashes of everything slot b's sessions hold for its first `upto` tokens, per part (GDN state, PLE history,
// indexer tails, dead rows, block positions, K, V, scales, pooled rows), for comparing a slot's state window by
// window against another run of the same request
bool QfnRuntime::slot_digest(int b, int64_t upto, uint64_t* part) {
    for (int i = 0; i < 10; ++i) part[i] = 1469598103934665603ull;
    auto mix = [&](int i, const uint8_t* p, size_t n) {
        uint64_t h = part[i];
        size_t k = 0;
        for (; k + 8 <= n; k += 8) {
            uint64_t w;
            std::memcpy(&w, p + k, 8);
            h = (h ^ w) * 1099511628211ull;
            h ^= h >> 29;
        }
        for (; k < n; ++k) h = (h ^ p[k]) * 1099511628211ull;
        part[i] = h;
    };
    std::string e;
    const basalt::core::SessionState& from = *bslot_ss[0][(size_t) b];
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    basalt::core::ConversationCheckpoint ck;
    ck.ids.assign((size_t) upto, 0);
    if (!basalt::core::conversation_checkpoint_save(ck, from, g, e)) return false;
    const basalt::core::CheckpointBytes* cv[5] = {&ck.state_a, &ck.state_b, &ck.state_c, &ck.state_d, &ck.state_e};
    for (int i = 0; i < 5; ++i) mix(i, cv[i]->data(), cv[i]->size());
    for (int64_t j = 0; j < from.qsa_alloc; ++j) {
        basalt::core::ConversationKv img;
        if (!basalt::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto & ~int64_t(3), true, e)) return false;   // whole pages: the cells past `upto` may hold rejected rows
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        const basalt::core::ConversationBuffer* kb[5] = {&img.k, &img.v, &img.k_scale, &img.v_scale, &img.pooled};
        for (int i = 0; i < 5; ++i) {   // (the pooled rows past the completed blocks are not state)
            const size_t cnt = i == 4 ? std::min(kb[i]->size(), (size_t) (upto >> 2) * (size_t) img.idx_dim * 4) : kb[i]->size();
            kb[i]->visit(0, cnt, [&](const uint8_t* p, size_t n, size_t) { mix(5 + i, p, n); return true; });
        }
    }
    return true;
}

}  // namespace basalt::models::qwen4exp::serve
