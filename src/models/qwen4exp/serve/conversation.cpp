// src/models/qwen4exp/serve/conversation.cpp - conversation operations for the persistent engine.
#include "basalt/core/diag.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "basalt/protocol.hpp"
#include "qfn_state.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#if defined(__linux__)
#include <sys/vfs.h>
#endif

namespace basalt::models::qwen4exp::serve {

ConvStateSizes conv_state_sizes(const basalt::core::ModelGeometry& g, const basalt::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    basalt::core::conversation_session_sizes(g, ss, z, error);
    return z;
}

/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint save: %s", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint restore: %s", error.c_str());
    return false;
}

/// The identity parked conversations on disk are written under (conversation_store.hpp): the model file
/// (its size and samples of its bytes) and the settings that change a state's bytes - the K/V type, the rope, the draft
/// layer, the image encoder, control vectors.  A file of another identity is never restored.
uint64_t conversation_identity(const Options& o, const basalt::core::ModelGeometry& g) {
    using basalt::core::conversation_store_hash;
    char num[512];
    std::snprintf(num, sizeof num, " rope %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g cvec %d %d %d %d", o.rope_scale,
                  o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor, o.yarn_attn_factor,
                  o.yarn_beta_fast, o.yarn_beta_slow, o.cvec_first, o.cvec_last, o.cvec_mode, o.cvec_single);
    std::string text = std::string("basalt conversation state kv ") + o.kv + " rope " +
                       o.rope_scaling + num + " mtp " + o.mtp + " vision_in_engine " + (o.vision_in_engine ? "1" : "0");
    if (o.kv == "int8" && basalt::core::exp_flag("int8_kv_rot") != 0) {
        text += " int8_kv_rot 1";
        const int width = basalt::kernels::kv_value_rotation_width(true);
        if (width != 256) text += " int8_kv_v_rot_width " + std::to_string(width);
    }
    for (const auto& [path, scale] : o.cvec_files) text += " cvec " + path + " " + std::to_string(scale);
    uint64_t h = conversation_store_hash(text.data(), text.size(), 1);
    const int64_t geo[] = {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_v_heads, g.n_head_kv,
                           g.head_dim, g.idx_key_dim, g.n_expert, g.n_ff};
    h = conversation_store_hash(geo, sizeof geo, h);
    return basalt::core::conversation_store_hash_file(o.model, h);
}

// finishes the in-flight host copy; a failed one drops its checkpoint
void QfnRuntime::ckd_finish() {
    const void* dst = ckd.pending();
    if (dst == nullptr) return;
    std::string ce;
    if (ckd.finish(ce)) return;
    basalt::log::error("basalt serve", "%s", ce.c_str());
    checks.erase(std::remove_if(checks.begin(), checks.end(),
                                [&](const ConvCheckpoint& k) { return k.state_a.data() == dst; }), checks.end());
}


// A capture of a conversation's state, or a read of one from disk, takes `need` bytes of RAM: only while the RAM
// guard reads its level as ok, and with `floor` left free after it (the disk tier's floor sits above the guard's).
bool QfnRuntime::capture_admit(uint64_t need, uint64_t floor) {
    if (need > UINT64_MAX - frozen_recovery_bytes) return false;
    return basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok &&
           basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(),
                                                   need + frozen_recovery_bytes, floor);
}

namespace {

bool complete_frozen_image(basalt::core::ConversationSharedSnapshot& job, std::string& error) {
    if (job.sources.empty()) return true;
    basalt::core::SavedConversation image;
    if (!basalt::core::conversation_snapshot_read_shared_draft(job, error) ||
        !basalt::core::conversation_snapshot_read_shared_index(job, error) ||
        !basalt::core::conversation_snapshot_finish_shared(image, std::move(job), error)) return false;
    job.image = std::move(image);
    job.sources.clear();
    job.pages.clear();
    return true;
}

}  // namespace

bool QfnRuntime::has_cold_profile_source() {
    if (!basalt::core::exp_flag("cold_shared_recovery") || !rope_context_profiles ||
        !shared_pages || !slot0_solo() || pipe_on || piped || !stages.empty() || batch_on() ||
        frozen_recovery.valid() || rsp != &ss || !live_ok || !live_imgs.empty() || bs.empty() ||
        bs[0].cached || bs[0].partial || shared_sequences.empty() || !shared_sequences[0] || !conversations->enabled() ||
        !recovery_stores[active_cache].enabled() || conversations->retained_reuse(&ss) ||
        live.size() < size_t(std::max<int64_t>(disk_min_tokens(), o.conversation_cache_min_tokens)))
        return false;
    const auto page = basalt::kernels::qsa_real_shapes().page_size;
    const auto& source = *shared_sequences[0];
    const uint32_t source_pages = static_cast<uint32_t>(1 + (live.size() - 1) / page);
    return cached_context_key(0) == active_cache &&
        basalt::core::SharedKvTable::covers_prefix(source.lease, source.stages, source_pages);
}

bool QfnRuntime::defer_profile_park(int slot, int64_t cells) {
    if (!has_cold_profile_source() || slot <= 0 || (size_t) slot >= shared_sequences.size() ||
        !shared_sequences[(size_t) slot]) return false;
    const auto page = basalt::kernels::qsa_real_shapes().page_size;
    const auto needed = static_cast<uint64_t>(1 + (cells - 1) / page);
    if (needed > shared_pages->available()) return false;
    const auto& source = *shared_sequences[0];

    auto& dest = recovery_stores[active_cache];
    const uint64_t cap = static_cast<uint64_t>(o.conversation_cache_mib) << 20;
    const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
    if (!dest.can_queue(0, cap) || disk_has(live, live_imgs, checks, cvec_cached, &dest)) return false;

    const auto t0 = Clock::now();
    ckd_finish();
    std::string why;
    try {
        const basalt::kernels::RopeContextScope outgoing(source.rope_context);
        const basalt::core::ConversationView view{live, live_imgs, checks, cvec_cached};
        size_t bytes = 0;
        if (!basalt::core::conversation_snapshot_bytes(view, ss, g, mtp_kv, bytes, why) ||
            !dest.can_queue(bytes, cap) || !capture_admit(bytes, floor)) return false;
        if (mtp_kv && !mtp.snapshot_main(why)) {
            basalt::log::warn("basalt serve", "cold shared snapshot: draft seal refused (%s)", why.c_str());
            return false;
        }
        basalt::core::SavedConversation chain;
        chain.live.ids = live;
        chain.live.imgs = live_imgs;
        chain.cvec = cvec_cached;
        chain.checkpoints.reserve(checks.size());
        for (auto& checkpoint : checks) chain.checkpoints.push_back(checkpoint.shared_copy());
        basalt::core::ConversationSharedSnapshot job;
        if (!basalt::core::conversation_snapshot_seal_shared(job, std::move(chain), ss, g, mtp_kv, why, true, cells)) {
            basalt::log::warn("basalt serve", "cold shared snapshot: seal refused (%s)", why.c_str());
            return false;
        }

        const auto tail_first = job.draft_tail.first_cell;
        const auto tail_end = job.draft_tail.end_cell;
        std::promise<bool> draft_done;
        auto draft_fence = draft_done.get_future().share();
        const auto started = Clock::now();
        frozen_recovery = std::async(std::launch::async, [job = std::move(job), store = &dest, cap, floor, started,
                                                         draft_done = std::move(draft_done)]() mutable {
            std::string error;
            bool queued = false, draft_signalled = false;
            auto copied = started;
            try {
                const bool draft_ready = basalt::core::conversation_snapshot_read_shared_draft(job, error);
                draft_done.set_value(draft_ready);
                draft_signalled = true;
                const bool complete = draft_ready && complete_frozen_image(job, error);
                copied = Clock::now();
                if (complete &&
                    basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), 0, floor)) {
                    auto writer = job.image.shared_recovery_view();
                    queued = writer && store->put(std::move(*writer), cap);
                }
            } catch (const std::exception& exception) {
                if (!draft_signalled) draft_done.set_value(false);
                basalt::log::warn("basalt serve", "cold shared snapshot worker: %s", exception.what());
            }
            if (!error.empty()) basalt::log::warn("basalt serve", "cold shared snapshot worker: %s", error.c_str());
            if (job.sources.empty())
                basalt::log::info("basalt serve", "cold shared snapshot: copied %zu tokens off-thread in %.1f ms; recovery queued=%d",
                                  job.image.live.ids.size(), std::chrono::duration<double, std::milli>(copied - started).count(),
                                  queued ? 1 : 0);
            return std::move(job);
        });
        if (tail_end) mtp.protect_draft_tail(tail_first, tail_end, std::move(draft_fence));
        frozen_recovery_bytes = bytes;
        frozen_recovery_cache = active_cache;
        std::printf("INFO frozen_shared_kv_pages=%zu frozen_shared_kv_cache=%u\n",
                    source.lease.pages().size(), active_cache);
        std::fflush(stdout);
        basalt::log::info("basalt serve", "cold shared snapshot: sealed %zu tokens in %.1f ms; source slot held",
                          live.size(), std::chrono::duration<double, std::milli>(started - t0).count());
        return true;
    } catch (const std::exception& exception) {
        basalt::log::warn("basalt serve", "cold shared snapshot: using synchronous park (%s)", exception.what());
        return false;
    }
}

void QfnRuntime::finish_frozen_recovery() {
    if (!frozen_recovery.valid()) return;
    const auto reserve = std::exchange(frozen_recovery_bytes, 0);
    try {
        auto job = frozen_recovery.get();   // physical source has not been retired
        mtp.protect_draft_tail(0, 0, {});
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
        std::string error;
        if (!job.sources.empty()) {
            const auto owned = job.image.bytes();
            if (!capture_admit(reserve > owned ? reserve - owned : 0, floor))
                error = "physical RAM admission refused the retry";
            else
                complete_frozen_image(job, error);
        }
        if (!job.sources.empty()) {
            basalt::log::warn("basalt serve", "cold shared snapshot: unfinished capture (%s)", error.c_str());
            return;
        }
        auto* cache = frozen_recovery_cache == active_cache ? &*conversations :
            parked_conversations[frozen_recovery_cache] ? &*parked_conversations[frozen_recovery_cache] : nullptr;
        if (cache && capture_admit(0, floor) && cache->put(std::move(job.image))) {
            auto& disk = disk_store(frozen_recovery_cache);
            for (auto& evicted : cache->take_spilled())
                if (!disk_has(evicted.live.ids, evicted.live.imgs, evicted.checkpoints, evicted.cvec, &disk))
                    disk.put(std::move(evicted), kDiskQueueCap);
        }
    } catch (const std::exception& exception) {
        mtp.protect_draft_tail(0, 0, {});
        basalt::log::warn("basalt serve", "cold shared snapshot: parked image unavailable (%s)", exception.what());
    }
}

bool QfnRuntime::before_command() {
    if (!frozen_recovery.valid()) return true;
    // BGEN's validated slot is fenced at publication. BGROW creates a new
    // table and session without writing the existing shared host pages.
    if ((line.starts_with("BGEN ") || line == "BGROW") &&
        frozen_recovery.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return true;
    return settle_shared_slot(0, false);
}

// Save only on a switch/rewind, not on each continuing request. No graph
// addresses change: all parked images live in ordinary host vectors.
// `kept`: the tokens of the outgoing conversation the session goes on holding (the request continues from its
// own chain), 0 when another conversation takes the session
bool QfnRuntime::park_current(size_t held, int64_t kept, bool profile_switch) {
    ckd_finish();
    if ((!conversations->enabled() && !disk_store().enabled()) || !live_ok || live.empty()) return true;
    // below the token minimum: store nothing, evict nothing
    if (o.conversation_cache_min_tokens > 0 && (int64_t) live.size() < o.conversation_cache_min_tokens) {
        basalt::log::info("basalt serve", "conversation cache: %zu tokens below the minimum %d, not parked",
                          live.size(), o.conversation_cache_min_tokens);
        return true;
    }
    // the disk tier takes the image when there is no RAM tier or it cannot make room, unless the conversation is
    // short or stored already, or the request goes on from the chain's deepest checkpoint (the same
    // conversation's next turn: all it drops is the last reply)
    bool to_disk = !conversations->enabled();
    int64_t deepest = 0;
    for (const ConvCheckpoint& c : checks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
    const bool disk_wants = disk_store().enabled() && !(kept > 0 && kept >= deepest) &&
                            !disk_has(live, live_imgs, checks, cvec_cached);
    auto& recovery = recovery_stores[active_cache];
    const uint64_t recovery_cap = static_cast<uint64_t>(o.conversation_cache_mib) << 20;
    const bool recovery_wants = profile_switch && stages.empty() &&
        recovery.can_queue(0, recovery_cap) && !disk_has(live, live_imgs, checks, cvec_cached, &recovery);
    if (to_disk && !disk_wants && !recovery_wants) return true;
    // before make_room evicts oldest-first, the copies of this conversation a turn back go (they hold nothing
    // the outgoing chain does not, apart from the tail this conversation rewrote)
    if (const size_t dropped = conversations->drop_superseded(live, live_imgs, checks, cvec_cached))
        basalt::log::info("basalt serve", "conversation cache: dropped %zu superseded cop%s of this "
                     "conversation; parked=%zu", dropped, dropped == 1 ? "y" : "ies", conversations->size());
    // A layer split parks one image per stage: the first stage's (with the draft layer's K/V, the drafter
    // runs on CUDA0) and one per later stage (without it).  The checkpoints are moved apart into their
    // stage parts for the capture and put back together whichever way this ends (no running state copied).
    const size_t n_st = stages.size();
    const basalt::core::QsaState* draft0 = mtp_kv;
    auto draft_of = [&](size_t) -> const basalt::core::QsaState* { return nullptr; };
    basalt::core::ConversationCheckpointSplit cs;
    if (n_st > 0) cs = basalt::core::conversation_checkpoints_split(std::move(checks), n_st);
    struct MergeBack {
        basalt::core::ConversationCheckpointSplit& cs; std::vector<ConvCheckpoint>& checks; bool on;
        ~MergeBack() { if (on && !basalt::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
    } merge_back{cs, checks, n_st > 0};
    const basalt::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
    auto complete = profile_switch && n_st == 0
        ? conversations->take_complete(rsp, dispatch_epoch, live, live_imgs, cvec_cached)
        : std::nullopt;
    auto reuse = conversations->take_reuse(rsp);
    std::vector<basalt::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
    stage_reuse.resize(n_st);
    reuse.stages.clear();
    // the later stages' estimates, each with its own retained K/V when it still fits
    size_t stage_estimate = 0, stage_retained = 0;
    for (size_t k = 0; k < n_st; ++k) {
        auto& st = stages[k];
        const basalt::core::OnDevice on(st->dev);
        const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
        size_t b = 0;
        if (!basalt::core::conversation_snapshot_bytes(view_k, st->ss, g, draft_of(k), b, err)) {
            basalt::log::warn("basalt serve", "conversation cache: skip parking (stage CUDA%d: %s)",
                         st->dev, err.c_str());
            err.clear();
            return true;
        }
        if (!stage_reuse[k].kv.empty()) {
            size_t r = 0;
            if (basalt::core::conversation_snapshot_capture_bytes(stage_reuse[k], view_k, st->ss, g, draft_of(k),
                                                                  r, err)) {
                b = r;
                stage_retained += stage_reuse[k].bytes();
            } else {
                stage_reuse[k] = {};
                err.clear();
            }
        }
        stage_estimate += b;
    }
    size_t estimate = complete ? complete->bytes() : 0;
    if (!complete && !basalt::core::conversation_snapshot_bytes(view, *rsp, g, draft0, estimate, err)) {
        basalt::log::warn("basalt serve", "conversation cache: skip parking (%s)", err.c_str());
        err.clear(); // A recoverable miss must not poison the batched draft prefill's error channel.
        return true;
    }
    const size_t fresh_estimate = estimate;
    if (!reuse.kv.empty() && !basalt::core::conversation_snapshot_capture_bytes(
            reuse, view, *rsp, g, draft0, estimate, err)) {
        reuse = {};
        estimate = fresh_estimate;
        err.clear();
    }
    estimate += stage_estimate;
    // A park carrying its own retained K/V replaces memory the cache
    // already held, so capacity is make_room's call - it runs next either
    // way, and put()'s accounting still bounds the budget. The with-reuse
    // estimate must stay uncapped: it counts the retained buffers'
    // capacity and directories, and put() charges that same true size -
    // a capped figure would under-evict and overfill the budget.
    const bool borrow = basalt::core::exp_flag("park_evicted_kv") != 0 &&
                        n_st == 0 && o.batch <= 1 && !disk_store().enabled() && live.size() >= 32768;
    basalt::core::ConversationKvReuse evicted;
    const bool ram_room = !to_disk && conversations->make_room(estimate, held, rsp,
                          borrow && reuse.kv.empty() ? &evicted : nullptr);
    if (!evicted.kv.empty()) {
        evicted.unchanged_tokens = std::min<int64_t>(evicted.unchanged_tokens, live.size());
        size_t donor_estimate = 0;
        if (basalt::core::conversation_snapshot_capture_bytes(evicted, view, *rsp, g, draft0, donor_estimate, err) &&
            donor_estimate <= estimate) {
            basalt::log::info("basalt serve", "park eviction donor covers %lld unchanged tokens",
                              (long long) evicted.unchanged_tokens);
            reuse = std::move(evicted);
            estimate = donor_estimate;
        } else {
            err.clear();
            evicted = {};
        }
    }
    if (!to_disk && !ram_room && disk_wants) to_disk = true;   // larger than the RAM tier: the disk tier takes it
    const bool disk_room = to_disk && disk_wants && disk_store().can_queue(estimate, kDiskQueueCap);
    const bool recovery_only = !ram_room && !disk_room && recovery_wants &&
        recovery.can_queue(estimate, recovery_cap);
    if (!ram_room && !disk_room && !recovery_only) {
        basalt::log::info("basalt serve", "conversation cache: skip parking (snapshot %zu MiB exceeds available budget)",
                     estimate >> 20);
        return true;
    }
    const auto t0 = Clock::now();
    try {
        if (ram_room && borrow && reuse.kv.empty() && basalt::core::exp_flag("park_share_kv") != 0) {
            auto shared = conversations->share_donor(rsp);
            if (!shared.kv.empty()) {
                shared.unchanged_tokens = std::min<int64_t>(shared.unchanged_tokens, live.size());
                size_t donor_estimate = 0;
                if (basalt::core::conversation_snapshot_capture_bytes(shared, view, *rsp, g, draft0, donor_estimate, err) &&
                    (donor_estimate <= estimate || conversations->can_fit(donor_estimate, held))) {
                    basalt::log::info("basalt serve", "park retained donor shares %lld unchanged tokens",
                                      (long long) shared.unchanged_tokens);
                    reuse = std::move(shared);
                    estimate = donor_estimate;
                } else {
                    err.clear();
                }
            }
        }
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024,
                                                  disk_store().enabled() ? disk_floor : 0);
        const size_t retained = complete ? estimate : reuse.bytes(true) + stage_retained;
        const size_t additional = estimate > retained ? estimate - retained : 0;
        if (!capture_admit(additional, floor)) {
            basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM admission; need %zu MiB plus %lld MiB floor, or telemetry unavailable, or RAM below the guard)",
                         additional >> 20, (long long) o.conversation_cache_min_free_mib);
            return true;
        }
        basalt::core::SavedConversation image;
        size_t reused_bytes = 0;
        const bool move_checks = !complete && !recovery_only && kept == 0 && held == 0 && n_st == 0 && o.batch <= 1 &&
            !disk_store().enabled() && live.size() <= 4096 && !checks.empty() &&
            basalt::core::exp_flag("park_move_short_checkpoints") != 0;
        if (complete) {
            image = std::move(*complete);
            for (const auto& layer : image.kv) reused_bytes += layer.bytes();
        } else {
            if (draft0 != nullptr && !mtp.snapshot_main(err)) return false;   // a streamed draft ring's own window
            const std::vector<ConvCheckpoint> none;
            const basalt::core::ConversationView capture{view.ids, view.images, move_checks ? none : view.checkpoints, view.cvec};
            if (!basalt::core::conversation_snapshot_save(image, capture, *rsp, g, draft0, err,
                    std::move(reuse), &reused_bytes)) return false;
        }
        for (size_t k = 0; k < n_st; ++k) {   // the later stages
            auto& st = stages[k];
            const basalt::core::OnDevice on(st->dev);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "stage sync failed"; return false; }
            const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
            basalt::core::SavedConversation part;
            if (!basalt::core::conversation_snapshot_save(part, view_k, st->ss, g, draft_of(k), err,
                    std::move(stage_reuse[k]), &reused_bytes))
                return false;
            image.stage_images.push_back(std::move(part));
        }
        if (!basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), 0, floor)) {
            basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)");
            return true;
        }
        const size_t moved_checkpoints = move_checks ? checks.size() : 0;
        if (move_checks) image.checkpoints.swap(checks);
        const size_t snapshot_bytes = image.bytes();
        bool stored = false;
        bool recovery_queued = false;
        if (ram_room && recovery_wants && recovery.can_queue(snapshot_bytes, recovery_cap)) {
            try {
                auto recovery_image = image.shared_recovery_view();
                recovery_queued = recovery_image && recovery.put(std::move(*recovery_image), recovery_cap);
            } catch (const std::bad_alloc&) {
                basalt::log::warn("basalt serve", "conversation cache: recovery sharing unavailable; using RAM park");
            }
        }
        try {
            if (recovery_only) stored = recovery_queued = recovery.put(std::move(image), recovery_cap);
            else if (disk_room) stored = disk_store().put(std::move(image), kDiskQueueCap);
            else stored = conversations->put(std::move(image), held);
        } catch (...) {
            if (move_checks) checks.swap(image.checkpoints);
            throw;
        }
        if (recovery_queued) {
            basalt::log::info("basalt serve", "conversation cache: queued %zu parked tokens for RAM recovery (handoff)",
                              live.size());
        }
        if (!stored && move_checks) checks.swap(image.checkpoints);
        if (stored && move_checks)
            basalt::log::info("basalt serve", "park moved %zu existing short checkpoints", moved_checkpoints);
        if (stored && borrow && !to_disk) conversations->remember_park(rsp, kept);
        basalt::log::info("basalt serve", "conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu%s%s",
                     stored ? "parked" : "skipped", live.size(),
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     conversations->size(), conversations->bytes(), conversations->evictions(), snapshot_bytes, reused_bytes,
                     recovery_only ? " (to RAM recovery)" : disk_room ? " (to disk)" : "",
                     complete ? " (completed recovery image)" : "");
        report_conversations();
        // what the RAM tier evicted to make room goes to the disk tier
        for (auto& s : conversations->take_spilled())
            if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec)) disk_store().put(std::move(s), kDiskQueueCap);
    } catch (const std::bad_alloc&) {
        // The active state has not been touched. Continue with normal
        // prompt processing rather than killing a serving process.
        basalt::log::warn("basalt serve", "conversation cache: allocation failed; skip parking");
    }
    return true;
}

bool QfnRuntime::gpu_sync_ok() {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) return true;
    ckpt_why = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
               ") - a GPU hang";
    return false;
}

// a checkpoint of the state after `cur[0, L)`, taken for `source`; false only when the copy itself failed.  `parts`: the stages'
// states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
bool QfnRuntime::checkpoint_at(int64_t L, std::vector<ConvCheckpoint>* parts, bool defer, ConvCheckpoint::Source source) {
    ckpt_why.clear();
    ckd_finish();
    if (o.prompt_cache <= 0 || L < 1) return true;
    for (ConvCheckpoint& c : checks)
        if ((int64_t) c.ids.size() == L) {   // a turn end or the root outranks a stride that landed on it
            c.used = ++check_clock;
            if (source > c.source) c.source = source;
            return true;
        }
    ConvCheckpoint c;
    c.source = source;
    c.ids.assign(cur.begin(), cur.begin() + L);
    c.imgs = imgs_below(req_imgs, L);
    if (parts != nullptr) {
        if (parts->size() != stages.size() + 1) return false;
        c.state_a = std::move((*parts)[0].state_a);
        c.state_b = std::move((*parts)[0].state_b);
        c.state_c = std::move((*parts)[0].state_c);
        c.state_d = std::move((*parts)[0].state_d);
        c.state_e = std::move((*parts)[0].state_e);
        for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
    } else {
        if (!gpu_sync_ok()) return false;
        std::string ce;
        if (defer && stages.empty()) {
            if (!ckd.save(c, *rsp, g, ce)) { basalt::log::error("basalt serve", "checkpoint save: %s", ce.c_str()); return false; }
        } else if (!checkpoint_save(c, *rsp, g)) return false;
        for (auto& st : stages) {   // a layer split's later stages: their sessions' part
            const basalt::core::OnDevice on(st->dev);
            ConvCheckpoint part;
            part.ids = c.ids;
            if (!gpu_sync_ok() || !checkpoint_save(part, st->ss, g)) return false;
            c.stage_parts.push_back(std::move(part));
        }
    }
    c.used = ++check_clock;
    checks.push_back(std::move(c));
    while ((int) checks.size() > o.prompt_cache) {
        std::vector<uint64_t> stamps;
        stamps.reserve(checks.size());
        for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
        const size_t victim = basalt::program::conv_cache::eviction_victim(stamps.data(), stamps.size(),
                                                                           o.prompt_cache);
        if (checks[victim].state_a.data() == ckd.pending()) ckd_finish();
        checks.erase(checks.begin() + (std::ptrdiff_t) victim);
    }
    return true;
}

unsigned QfnRuntime::cached_adapter(int slot) const {
    return slot >= 0 && shared_pages ? shared_sequences[(size_t) slot]->adapter : active_adapter;
}

unsigned QfnRuntime::cached_context_key(int slot) const {
    if (slot < 0 || !shared_pages) return active_cache;
    const auto& sequence = *shared_sequences[(size_t) slot];
    return cache_key(sequence.adapter, sequence.rope_context);
}

uint64_t QfnRuntime::adapter_cache_identity(unsigned mode, uint64_t identity) const {
    using basalt::core::conversation_store_hash;
    if (mode & 1u) {
        constexpr char version[] = "AtomicChat rank-one v1";
        identity = conversation_store_hash(version, sizeof version, identity);
        identity = conversation_store_hash(&atomic_lora_scale, sizeof atomic_lora_scale, identity);
        identity = lora_host->state_identity(identity);
    }
    if (mode & 2u) {
        constexpr char version[] = "Swift LoRA rank-32 v1";
        identity = conversation_store_hash(version, sizeof version, identity);
        identity = conversation_store_hash(&swift_lora_scale, sizeof swift_lora_scale, identity);
        identity = swift_host->state_identity(identity);
    }
    return identity;
}

// The conversations the engine holds go to the disk tier when no stored copy reaches within kDiskStaleTokens of
// them: the RAM tier's (at exit), the solo session's and every idle slot's, each with its draft layer's K/V.  Between
// requests (no slot decoding) this bounds what a crash or a kill loses; a slot's draft K/V is copied into the main
// store for the capture, which every request refills from its own store first, so between requests that runs only
// when the solo session holds no conversation of its own there.  At most `limit` images; returns those queued.
size_t QfnRuntime::spill_to_disk(bool at_exit, size_t limit) {
    if (at_exit && frozen_recovery.valid() && !settle_shared_slot(0, false)) {
        basalt::log::warn("basalt serve", "cold shared snapshot: source retirement at exit failed (%s)", err.c_str());
        return 0;
    }
    // With a RAM tier, persist on eviction or fallback instead of capturing the
    // live session again every time it goes idle. Shutdown still drains the cache.
    if (!at_exit && conversations->enabled()) return 0;
    return spill_conversations(at_exit, limit, false);
}

size_t QfnRuntime::checkpoint_for_recovery() {
    observe_scheduler(SchedulerPhase::idle);
    if (frozen_recovery.valid()) {
        if (frozen_recovery.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return 0;
        if (!settle_shared_slot(0, false)) return 0;
    }
    if (!recovery_stores[0].enabled() ||
        std::any_of(recovery_stores.begin(), recovery_stores.end(),
                    [](const auto& store) { return store.queued_bytes() != 0; })) return 0;
    return spill_conversations(false, 1, true);
}

size_t QfnRuntime::spill_conversations(bool at_exit, size_t limit, bool ram_recovery) {
    auto& retry_at = idle_spill_retry[ram_recovery];
    if (!at_exit && Clock::now() < retry_at) return 0;
    bool ram_denied = false;
    auto store_for = [&](unsigned mode) -> basalt::core::ConversationStore& {
        return ram_recovery ? recovery_stores[mode] : disk_store(mode);
    };
    const basalt::core::ConversationCaptureStop stop = [this, stopping = stop_req.load()] {
        std::lock_guard<std::mutex> lock(in_mu);
        return in_eof || !in_lines.empty() || (!stopping && stop_req.load());
    };
    const bool idle_recovery = ram_recovery && !at_exit;
    if (idle_recovery && stop()) return 0;
    ckd_finish();   // a turn checkpoint's state may still be on its way to the host
    size_t written = 0;
    const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
    // Shutdown flushes before every submission. A single admitted image can
    // use the configured disk budget without enlarging the live queue.
    const uint64_t exit_cap = at_exit && o.conversation_cache_disk_mib > 0
        ? std::max(kDiskQueueCap, static_cast<uint64_t>(o.conversation_cache_disk_mib) << 20)
        : kDiskQueueCap;
    if (at_exit) {
        const auto drain = [&](basalt::core::ConversationCache& cache, basalt::core::ConversationStore& dest) {
            while (dest.enabled() && cache.size() > 0) {
                basalt::core::SavedConversation s = cache.take(cache.size() - 1);
                if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec, &dest)) {
                    dest.flush();
                    const bool large = exit_cap > kDiskQueueCap && s.bytes() > kDiskQueueCap;
                    if (dest.put(std::move(s), exit_cap)) {
                        ++written;
                        if (large) dest.flush();   // finish before another namespace can submit
                    }
                }
            }
        };
        drain(*conversations, disk_store());
        for (unsigned mode = 0; mode < cache_count; ++mode)
            if (parked_conversations[mode]) drain(*parked_conversations[mode], disk_store(mode));
    }
    const uint64_t queue_cap = ram_recovery
        ? static_cast<uint64_t>(o.conversation_cache_mib) << 20 : exit_cap;
    auto spill = [&](unsigned mode, basalt::core::ConversationStore& dest, const basalt::core::SessionState& s,
                     const std::vector<int32_t>& ids,
                     const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
        if (idle_recovery && stop()) return;
        const basalt::core::ConversationView view{ids, imgs, cks, cvec};
        size_t bytes = 0;
        std::string se;
        if (!basalt::core::conversation_snapshot_bytes(view, s, g, mtp_kv, bytes, se)) {
            basalt::log::warn("basalt serve", "conversation cache: not written to disk (%s)", se.c_str());
            return;
        }
        if (at_exit) dest.flush();   // one image in RAM at a time
        auto* cache = mode == active_cache ? &*conversations
                    : parked_conversations[mode] ? &*parked_conversations[mode] : nullptr;
        const auto* retained = ram_recovery && cache ? cache->retained_reuse(&s) : nullptr;
        size_t additional = bytes;
        if (retained && retained->stages.empty()) {
            size_t peak = 0;
            if (basalt::core::conversation_snapshot_capture_bytes(*retained, view, s, g, mtp_kv, peak, se)) {
                bytes = peak;
                const auto already_held = retained->bytes(true);
                additional = peak > already_held ? peak - already_held : 0;
            } else {
                retained = nullptr;
                se.clear();
            }
        } else {
            retained = nullptr;
        }
        if (!dest.can_queue(bytes, queue_cap)) {
            if (at_exit && bytes > queue_cap)
                basalt::log::info("basalt serve", "conversation cache: %zu tokens exceed shutdown queue "
                                  "(snapshot %zu MiB, limit %llu MiB)", ids.size(), bytes >> 20,
                                  (unsigned long long) (queue_cap >> 20));
            return;
        }
        if (!capture_admit(additional, floor)) {
            if (at_exit || (!ram_denied && retry_at == Clock::time_point{}))
                basalt::log::info("basalt serve", "conversation cache: %zu tokens not queued for %s (physical RAM "
                                  "admission; snapshot %zu MiB, floor %llu MiB)",
                                  ids.size(), ram_recovery ? "RAM recovery" : "disk", additional >> 20,
                                  (unsigned long long) (floor >> 20));
            ram_denied = true;
            return;
        }
        try {
            const auto t0 = Clock::now();
            basalt::core::SavedConversation image;
            if (idle_recovery && stop()) return;
            if (mtp_kv != nullptr && !mtp.snapshot_main(se)) {
                basalt::log::warn("basalt serve", "conversation cache: draft snapshot failed (%s)", se.c_str());
                return;
            }
            if (idle_recovery && stop()) return;
            auto reuse = retained ? cache->take_reuse(&s) : basalt::core::ConversationKvReuse{};
            // An interrupted image is discarded. The committed recovery point remains available.
            const auto* interrupt = idle_recovery ? &stop : nullptr;
            size_t reused = 0;
            if (!basalt::core::conversation_snapshot_save(image, view, s, g, mtp_kv, se,
                                                          std::move(reuse), &reused, interrupt)) {
                if (se == basalt::core::kCaptureInterrupted) {
                    basalt::log::info("basalt serve", "conversation cache: idle recovery for %zu tokens yielded to input after %.0f ms",
                                      ids.size(), std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                } else
                    basalt::log::warn("basalt serve", "conversation cache: not written to disk (%s)", se.c_str());
                return;
            }
            std::optional<basalt::core::SavedConversation> writer;
            std::vector<basalt::core::ConversationKv> park_kv;
            if (ram_recovery && cache == &*conversations && cache->enabled() && &s == rsp && stages.empty()) {
                if (&s == &ss && &ids == &live && cache->can_retain_complete(image.bytes()))
                    writer = image.shared_recovery_view();
                if (!writer) {
                    park_kv.reserve(image.kv.size());
                    for (const auto& layer : image.kv) park_kv.push_back(layer.shared_copy());
                }
            }
            if (dest.put(writer ? std::move(*writer) : std::move(image), queue_cap)) {
                if (writer)
                    cache->retain_complete(std::move(image), &s, dispatch_epoch + 1);
                else if (!park_kv.empty())
                    cache->retain(std::move(park_kv), (int64_t) ids.size(), {}, &s);
                if (reused != 0)
                    basalt::log::info("basalt serve", "conversation cache: reused %zu MiB of recovery KV", reused >> 20);
                ++written;
                if (at_exit && bytes > kDiskQueueCap) {
                    dest.flush();
                    basalt::log::info("basalt serve", "conversation cache: shutdown queue drained for %zu tokens (%zu MiB captured)",
                                      ids.size(), bytes >> 20);
                }
                if (!at_exit)
                    basalt::log::info("basalt serve", "conversation cache: queued %zu tokens for %s "
                                      "(captured in %.0f ms)", ids.size(), ram_recovery ? "RAM recovery" : "disk",
                                      std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            }
        } catch (const std::bad_alloc&) {
            basalt::log::warn("basalt serve", "conversation cache: allocation failed; not written to disk");
        }
    };
    if (idle_recovery && stop()) return written;
    std::string se;
    if (mtp_on && !mtp.use_main(se)) {
        if (ram_recovery) return written;
        se.clear();
    }
    const bool slot0_cached = slot0_is_solo && !bs.empty() && bs[0].cached;
    const bool solo_holds = live_ok && !live.empty() && !slot0_cached;
    auto& solo_store = store_for(active_cache);
    if (solo_holds && solo_store.can_queue(0, queue_cap) &&
        !disk_has(live, live_imgs, checks, cvec_cached, &solo_store))
        spill(active_cache, solo_store, ss, live, live_imgs, checks, cvec_cached);
    const bool scan_slots = !solo_holds || at_exit || !batch_spec;   // preserve the solo draft K/V in the main store
    for (int b = 0; scan_slots && b < (int) bs.size() && written < limit; ++b) {
        if (idle_recovery && stop()) return written;
        BSlot& sl = bs[(size_t) b];
        if (!sl.cached || sl.active || sl.partial) continue;
        const unsigned mode = cached_context_key(b);
        auto& dest = store_for(mode);
        if (!dest.can_queue(0, queue_cap) || disk_has(sl.ids, sl.imgs, sl.checks, sl.cvec, &dest))
            continue;
        if (idle_recovery && stop()) return written;
        if (batch_spec && !mtp.copy_store(b, mtp.main_store(), (int64_t) sl.ids.size(), se)) {
            se.clear();
            continue;
        }
        make_view(rviews[(size_t) b], b);
        spill(mode, dest, rviews[(size_t) b], sl.ids, sl.imgs, sl.checks, sl.cvec);
    }
    if (at_exit) {
        for (unsigned mode = 0; mode < cache_count; ++mode) disk_store(mode).flush();
    } else {
        // Finish scanning before backing off so a smaller eligible snapshot can still be queued.
        retry_at = ram_denied && written == 0 ? Clock::now() + std::chrono::seconds(10) : Clock::time_point{};
    }
    return written;
}

namespace {
// the VRAM a session's own QSA layers' K/V maps (the drafter's arena aside)
uint64_t qsa_kv_mapped(const basalt::core::SessionState& u) {
    uint64_t b = 0;
    for (int64_t j = 0; u.qsa_states != nullptr && j < u.qsa_alloc; ++j)
        if (const basalt::core::KvArena* a = u.qsa_states[u.qsa_ord0 + j].kv_arena) b += a->mapped_bytes();
    return b;
}
}  // namespace

// The idle conversations whose KV may leave VRAM, the longest idle first (the busy clock when each went idle, and
// its batch slot, -1 for the solo session's): those whose time is up (kParkAfterBusy), or with `any` every one.
// Idle: a cached batch slot other than `park_keep`, or the solo session's conversation while no request works
// there, holding cells beyond the base; not session `s` (null: none is excluded).
std::vector<std::pair<double, int>> QfnRuntime::idle_list(const basalt::core::SessionState* s, bool any) {
    std::vector<std::pair<double, int>> idle;
    if (kv_floor <= 0 || bs.empty()) return idle;
    const double now = busy_now();
    auto holds = [&](const basalt::core::SessionState& u, double since) {
        return (s == nullptr || u.qsa_states != s->qsa_states) && (any || now - since >= kParkAfterBusy) &&
               basalt::core::session_kv_cells(u) > kv_floor;
    };
    for (int b = 0; b < (int) bs.size(); ++b) {
        const BSlot& sl = bs[(size_t) b];
        if (sl.cached && !sl.active && !sl.partial && b != park_keep && holds(*bslot_ss[0][(size_t) b], sl.idle_busy))
            idle.emplace_back(sl.idle_busy, b);
    }
    const bool solo = solo_out != nullptr ? solo_out->live_ok && !solo_out->live.empty()
                                          : !solo_busy && live_ok && !live.empty();
    if (solo && !(slot0_is_solo && (bs[0].cached || bs[0].active || park_keep == 0)) && holds(ss, solo_idle_busy))
        idle.emplace_back(solo_idle_busy, -1);
    std::sort(idle.begin(), idle.end());
    return idle;
}

// The VRAM session `s` is `need` bytes short of comes first from the KV idle conversations hold (idle_list), each
// moved out whole by park_one.  Returns the bytes given back, -1 when cells could not be given back.  The device is
// idle.
int64_t QfnRuntime::park_idle(const basalt::core::SessionState* s, int64_t need, bool any, std::string& e) {
    int64_t given = 0;
    for (const auto& idle : idle_list(s, any)) {
        if (given >= need) break;
        const int b = idle.second;
        const basalt::core::SessionState& u = b >= 0 ? *bslot_ss[0][(size_t) b] : ss;
        const uint64_t had = qsa_kv_mapped(u);
        if (!park_one(b, e)) {
            if (!e.empty()) return -1;
            continue;
        }
        given += (int64_t) (had - qsa_kv_mapped(u));
    }
    return given;
}

// An idle engine (no request, no slot decoding): one idle conversation whose time is up leaves VRAM and the expert
// cache maps what it held, while the cache is below its full size.  False when none went (`e` set on a failure).
bool QfnRuntime::park_idle_engine(std::string& e) {
    if (!xcache.segmented() || xcache.mapped_bytes() >= xcache.full_bytes() || idle_list(nullptr, false).empty())
        return false;
    if (!kv_drain(e)) return false;
    const int64_t before = xcache.slots();
    if (park_idle(nullptr, 1, false, e) <= 0 || !cache_regrow(e)) return false;
    basalt::log::info("basalt serve", "idle: the expert cache %lld -> %lld slots", (long long) before,
                      (long long) xcache.slots());
    return true;
}

// Idle conversation `b` (a batch slot; -1: the solo session's) out of VRAM, its K/V cells beyond the base back to the
// driver: into the RAM tier when the RAM a capture must leave admits it, else nothing is copied when the disk tier
// holds it.  False when it stays in VRAM (`e` empty) or its cells could not be given back (`e` set).
bool QfnRuntime::park_one(int b, std::string& e) {
    if (b <= 0 && frozen_recovery.valid()) return false;
    const auto t0 = Clock::now();
    basalt::core::SessionState& u = b >= 0 ? *bslot_ss[0][(size_t) b] : ss;
    std::vector<int32_t>& ids = b >= 0 ? bs[(size_t) b].ids : solo_out != nullptr ? solo_out->live : live;
    std::vector<ImgKey>& imgs = b >= 0 ? bs[(size_t) b].imgs : solo_out != nullptr ? solo_out->live_imgs : live_imgs;
    std::vector<ConvCheckpoint>& cks = b >= 0 ? bs[(size_t) b].checks : solo_out != nullptr ? solo_out->checks : checks;
    const bool cvec = b >= 0 ? bs[(size_t) b].cvec : solo_out != nullptr ? solo_out->cvec : cvec_cached;
    const std::string who = b >= 0 ? "slot " + std::to_string(b) : std::string("the solo session");
    basalt::core::SessionState* cap = &u;   // a slot's state is captured through its view of the solo session
    if (&u != &ss) {
        make_view(rviews[(size_t) b], b);
        cap = &rviews[(size_t) b];
    }
    const int store = b >= 0 ? b : solo_draft_in0 ? 0 : mtp.main_store();
    const unsigned mode = cached_context_key(b);
    auto& dest = disk_store(mode);
    auto& cache = mode == active_cache ? *conversations : *parked_conversations[mode];
    std::string se;
    const basalt::core::ConversationView view{ids, imgs, cks, cvec};
    size_t est = 0;
    if (!basalt::core::conversation_snapshot_bytes(view, *cap, g, mtp_kv, est, se)) {
        basalt::log::warn("basalt serve", "%s's idle conversation stays in VRAM (%s)", who.c_str(), se.c_str());
        return false;
    }
    size_t moved = 0;   // its checkpoints are in RAM already: they move into the image
    for (const ConvCheckpoint& c : cks) moved += c.bytes();
    // the RAM a capture leaves free: the conversation cache's, the disk tier's, and the reserve
    const uint64_t floor = std::max({(uint64_t) o.conversation_cache_min_free_mib << 20,
                                     dest.enabled() ? disk_floor : 0, basalt::platform::ram_reserve()});
    // below the token minimum: cells go back, the conversation is kept in neither tier
    const bool keep = o.conversation_cache_min_tokens <= 0 ||
                      (int64_t) ids.size() >= o.conversation_cache_min_tokens;
    bool to_ram = keep && capture_admit(est > moved ? est - moved : 0, floor);
    const bool on_disk = keep && !to_ram && dest.enabled() && (int64_t) ids.size() >= disk_min_tokens() &&
                         disk_has(ids, imgs, cks, cvec, &dest);
    if (!to_ram && !on_disk && keep) {   // said once per idle stretch
        double& noted = b >= 0 ? bs[(size_t) b].park_noted : solo_park_noted;
        const double since = b >= 0 ? bs[(size_t) b].idle_busy : solo_idle_busy;
        if (noted != since)
            basalt::log::info("basalt serve", "%s's idle conversation stays in VRAM: %zu MiB do not fit in RAM above "
                              "the %llu MiB floor", who.c_str(), (est - std::min(est, moved)) >> 20,
                              (unsigned long long) (floor >> 20));
        noted = since;
        return false;
    }
    size_t bytes = 0;
    if (to_ram) {
        try {
            basalt::core::SavedConversation image;
            const std::vector<ConvCheckpoint> none;
            const basalt::core::ConversationView bare{ids, imgs, none, cvec};
            if (!basalt::core::conversation_snapshot_save(image, bare, *cap, g,
                    static_cast<const basalt::core::QsaState*>(nullptr), se) ||
                (mtp_kv != nullptr && !mtp.save_store(store, (int64_t) ids.size(), image.kv.emplace_back(), se))) {
                basalt::log::warn("basalt serve", "%s's idle conversation stays in VRAM (%s)", who.c_str(), se.c_str());
                return false;
            }
            if (!basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), 0, floor)) {
                basalt::log::info("basalt serve", "%s's idle conversation stays in VRAM: RAM below the %llu MiB floor "
                                  "after its capture", who.c_str(), (unsigned long long) (floor >> 20));
                return false;
            }
            image.checkpoints = std::move(cks);
            bytes = image.bytes();
            cache.put_slot(std::move(image), bs.size());
            for (auto& sc : cache.take_spilled())
                if (!disk_has(sc.live.ids, sc.live.imgs, sc.checkpoints, sc.cvec, &dest))
                    dest.put(std::move(sc), kDiskQueueCap);
        } catch (const std::bad_alloc&) {
            basalt::log::warn("basalt serve", "%s's idle conversation stays in VRAM: allocation failed", who.c_str());
            return false;
        }
    }
    const int64_t tokens = (int64_t) ids.size(), had = basalt::core::session_kv_cells(u);
    if (b >= 0) bs[(size_t) b].cached = false;
    else if (solo_out != nullptr) solo_out->live_ok = false;
    else live_ok = false;
    ids.clear();
    imgs.clear();
    cks.clear();
    for (int64_t j = 0; j < u.qsa_alloc; ++j)
        if (basalt::core::KvArena* a = u.qsa_states[u.qsa_ord0 + j].kv_arena)
            if (!a->unmap(kv_floor, e)) { e = "elastic KV: " + e; return false; }
    ++kv_moves;
    char where[64];
    if (to_ram) std::snprintf(where, sizeof where, "to RAM (%.0f MiB)", (double) bytes / 1048576.0);
    else if (on_disk) std::snprintf(where, sizeof where, "kept on disk only");
    else std::snprintf(where, sizeof where, "dropped, below the %d token minimum", o.conversation_cache_min_tokens);
    basalt::log::info("basalt serve", "%s's idle conversation, %lld tokens, %s in %.0f ms: %lld KV cells back",
                      who.c_str(), (long long) tokens, where,
                      std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                      (long long) (had - basalt::core::session_kv_cells(u)));
    return true;
}

int QfnRuntime::init_conversations() {
    live_ok = false;

    check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it

    // (the checkpoints' pooled blocks were reserved, their pages written, before the RAM copy)
    if (o.prompt_cache > 0 && !multi_gpu) {
        std::string ce;
        if (!ckd.reserve(conv_state_sizes(g, ss).gdn, ce))
            basalt::log::warn("basalt serve", "deferred checkpoint copy off: %s", ce.c_str());
    }

    cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
    // the session the request path reads, parks and restores: the solo one, or (a --batch request for a slot other
    // than 0) a view of it carrying that slot's state, so the prompt is read in the slot itself
    // (one view per slot, each at a fixed address: the verifier keeps solo window graphs per view)
    rviews.resize((size_t) std::max(o.batch, 1));
    rsp = &ss;
    conversations.emplace(
        o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
        (size_t) o.conversation_cache_slots);

    disk_floor = 0;                              // the RAM a capture or a read leaves free (set below)

    constexpr std::array<const char*, adapter_count> adapter_dirs{"", "/atomic-chat", "/swift", "/atomic-chat-swift"};
    const auto base_identity = conversation_identity(o, g);
    std::array<uint64_t, cache_count> identities{};
    std::array<std::string, cache_count> directories;
    for (unsigned key = 0; key < cache_count; ++key) {
        if (!cache_configured(key)) continue;
        const unsigned mode = key % adapter_count;
        const bool yarn = key >= adapter_count;
        uint64_t identity = base_identity;
        if (rope_context_profiles) {
            const std::string_view label = yarn ? "flash-next rope yarn2 v1" : "flash-next rope native v1";
            identity = basalt::core::conversation_store_hash(label.data(), label.size(), identity);
            directories[key] = yarn ? "/rope-yarn2" : "/rope-native";
        }
        directories[key] += adapter_dirs[mode];
        identities[key] = adapter_cache_identity(mode, identity);
    }

    if (o.conversation_cache_disk_mib > 0) {
        std::string de;
        const auto t0 = Clock::now();
        disk_store().set_row_cells(basalt::kernels::qsa_real_shapes().idx_block);
        if (o.prompt_cache <= 0 || !stages.empty()) {
            basalt::log::warn("basalt serve", "conversation cache on disk off: %s",
                              o.prompt_cache <= 0 ? "--prompt-cache 0" : "a layer split");
        } else if (!disk_store().open(o.conversation_cache_dir + directories[0], identities[0],
                              (uint64_t) o.conversation_cache_disk_mib << 20, de)) {
            basalt::log::warn("basalt serve", "conversation cache on disk off: %s", de.c_str());
        } else {
            conversations->set_spill(true);
            const auto st = disk_store().stats();
            basalt::log::info("basalt serve", "conversation cache on disk: %s, %zu conversations of this model "
                              "(%.2f GiB), budget %lld MiB, indexed in %.0f ms", (o.conversation_cache_dir + directories[0]).c_str(),
                              st.files, (double) st.bytes / 1073741824.0, (long long) o.conversation_cache_disk_mib,
                              std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
    }

    std::vector<basalt::core::ConversationStore*> disk_group;
    if (disk.enabled()) disk_group.push_back(&disk);
    for (unsigned mode = 1; mode < cache_count; ++mode) {
        if (!cache_configured(mode)) continue;
        parked_conversations[mode].emplace(
            o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
            (size_t) o.conversation_cache_slots);
        if (!disk.enabled()) continue;
        std::string de;
        auto& store = disk_store(mode);
        store.set_row_cells(basalt::kernels::qsa_real_shapes().idx_block);
        if (!store.open(o.conversation_cache_dir + directories[mode], identities[mode],
                        (uint64_t) o.conversation_cache_disk_mib << 20, de)) {
            basalt::log::warn("basalt serve", "namespace %u disk cache off: %s", mode, de.c_str());
            continue;
        }
        disk_group.push_back(&store);
        parked_conversations[mode]->set_spill(true);
    }
    basalt::core::ConversationStore::share_budget(disk_group);
    if (disk_group.size() > 1)
        basalt::log::info("basalt serve", "conversation cache on disk: shared %lld MiB budget across %zu namespaces",
                          (long long) o.conversation_cache_disk_mib, disk_group.size());

    if (const char* directory = std::getenv("BASALT_RAM_RECOVERY_DIR"); directory && *directory) {
#if defined(__linux__)
        struct statfs fs{};
        if (o.shared_expert_arena.empty() || statfs(directory, &fs) != 0 ||
            static_cast<uint64_t>(fs.f_type) != 0x01021994 || o.prompt_cache <= 0 || o.conversation_cache_mib <= 0) {
            basalt::log::error("basalt serve", "RAM recovery needs an existing tmpfs directory, a shared expert arena, and a conversation cache");
            return 1;
        }
        const auto budget = static_cast<uint64_t>(o.conversation_cache_mib) << 20;
        std::vector<basalt::core::ConversationStore*> recovery_group;
        for (unsigned mode = 0; mode < cache_count; ++mode) {
            if (!cache_configured(mode)) continue;
            std::string re;
            auto& store = recovery_stores[mode];
            store.set_row_cells(basalt::kernels::qsa_real_shapes().idx_block);
            if (!store.open(std::string(directory) + directories[mode], identities[mode], budget, re)) {
                basalt::log::error("basalt serve", "RAM recovery for namespace %u: %s", mode, re.c_str());
                return 1;
            }
            recovery_group.push_back(&store);
        }
        basalt::core::ConversationStore::share_budget(recovery_group);
#else
        basalt::log::error("basalt serve", "RAM recovery requires Linux tmpfs");
        return 1;
#endif
    }

    pp_total = 0;
    pp_from = 0;
    pp_next_check = 0;
    // the end of the part being read when the part loop saves a checkpoint there (else -1): the periodic
    // checkpoint skips that cell, the loop's copy runs beside what follows
    pp_part_ckpt = -1;
    // the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled mid-read
    // reports as read, instead of the whole prompt
    pp_reached = 0;
    pp_t0 = Clock::now();

    part_next = std::vector<int64_t>(stages.size() + 1, INT64_MAX);

    // live progress inside a chunk: PL <estimated position> <prompt tokens> <ms> <tokens/s>, at most twice a second
    sp.on_layer = [&, last_ms = 0.0](int64_t layers_done, int64_t n_layers, int64_t T, int64_t p0) mutable {
        if (pp_total <= 0 || layers_done <= 0 || n_layers <= 0) return;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
        if (ms - last_ms < 500.0 && ms >= last_ms) return;
        last_ms = ms;
        const int64_t at = p0 + T * layers_done / n_layers;
        basalt::protocol::emit("PL %lld %lld %.0f %.1f\n", (long long) at, (long long) pp_total, ms,
                               ms > 0.0 ? 1000.0 * (double) (at - pp_from) / ms : 0.0);
    };
    sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
        std::vector<int32_t> nxt((size_t) T);
        for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
        // batched through the prompt path when it can (one GPU: a layer split's drafter is on the last stage)
        const bool batched = mtp_on && !multi_gpu && sp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e);
        if (!e.empty() || (mtp_on && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
        // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
        const int64_t done = p0 + T;
        pp_reached = done;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                    ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
        basalt::core::progress_at("reading the prompt (batched), done up to token", done);
        basalt::core::progress_beat();
        std::fflush(stdout);
        if (o.prompt_cache_every > 0 && done >= pp_next_check) {
            if (done == pp_part_ckpt && !multi_gpu) {
                pp_next_check = done + o.prompt_cache_every;
                return true;
            }
            bool saved = false;
            if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                std::vector<ConvCheckpoint> parts;
                {
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto it = part_at.find(done);
                    if (it != part_at.end()) parts = std::move(it->second);
                    part_at.erase(part_at.begin(), part_at.upper_bound(done));
                }
                bool complete = parts.size() == stages.size() + 1;
                for (const ConvCheckpoint& k : parts) complete = complete && !k.state_a.empty();
                saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
            } else {
                saved = checkpoint_at(done);
            }
            if (!saved) { e = "saving a conversation checkpoint failed" + ckpt_why; return false; }
            pp_next_check = done + o.prompt_cache_every;
        }
        return true;
    };
    if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
        stages.back()->sp.on_chunk = std::move(sp.on_chunk);
        sp.on_chunk = nullptr;
        for (size_t i = 0; i <= stages.size(); ++i) {
            basalt::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
            basalt::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
            stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                part_next[i] = done + o.prompt_cache_every;
                ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                part.ids.assign(cur.begin(), cur.begin() + done);
                if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                std::lock_guard<std::mutex> lk(part_mu);
                auto& v = part_at[done];
                v.resize(stages.size() + 1);
                v[i] = std::move(part);
                return true;
            };
        }
    }
    drive.d.plan = ver.plan_sink();
    drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
    return 0;
}

}  // namespace basalt::models::qwen4exp::serve
