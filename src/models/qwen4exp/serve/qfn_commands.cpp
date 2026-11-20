// src/models/qwen4exp/serve/qfn_commands.cpp - the protocol commands that reach Flash-Next's tiers.
#include "models/qwen4exp/adapter.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "basalt/vision/ffmpeg_api.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include <charconv>
#include <cstdlib>
#include <set>
#include <string_view>
#include <stdexcept>
#include "../../../program/tokenizer/unicode_flags.hpp"
#include "../../../program/cli/tokens.hpp"

namespace basalt::models::qwen4exp::serve {

bool QfnRuntime::windows_in_flight() { return piped && pipe_inflight(); }
bool QfnRuntime::next_batch_window() { return service_vision() && (piped ? pump(true) : batch_step()); }
// a request reads its prompt through every stage: the groups in flight finish first
bool QfnRuntime::drain_windows() { return !piped || pipe_drain(); }

int QfnRuntime::grow_batch_slot() {
    if (!shared_pages) return -1;
    if (line != "BGROW" &&
        (!has_cold_profile_source() || (line.back() == '1') == (active_cache >= adapter_count))) return -1;
    std::string error;
    const int slot = static_cast<int>(bs.size());
    if (append_shared_session(error)) return slot;
    basalt::log::warn("basalt serve", "shared slot growth: %s", error.c_str());
    return -1;
}

void QfnRuntime::script_command(const std::string& line) {
    char* e1 = nullptr;
    const long b = std::strtol(line.c_str() + 8, &e1, 10);
    char* e2 = nullptr;
    const unsigned long long seed = std::strtoull(e1, &e2, 10);
    std::vector<int64_t> sids;
    std::string se;
    if (b < 0 || b >= (long) bs.size() || e2 == e1 || !cli::parse_i64_list(e2, sids, se)) {
        std::printf("ERR BSCRIPT: bad slot or list\n");
        std::fflush(stdout);
        return;
    }
    bscript[(size_t) b].first.assign(sids.begin(), sids.end());
    bscript[(size_t) b].second = seed;
}

void QfnRuntime::probe_command(const std::string& line, std::string& perr) {
    if (!service_vision(true)) { perr = err; std::exit(1); }
if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K share=F: live values for the next requests
        {
            if (ba_job) { ba_thr.join(); ba_job = false; }
            std::istringstream kvs(line.substr(10));
            for (std::string kv; kvs >> kv;) {
                const size_t eq = kv.find('=');
                const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
                const double v = eq == std::string::npos ? -1.0 : std::atof(kv.c_str() + eq + 1);
                if (key == "workers" && v >= 1) pool.set_active_workers((int) v);
                else if (key == "share" && v >= 0 && v <= 1) sp.set_cpu_share(v);
                else perr = "PROBE: bad setting " + kv;
            }
            if (perr.empty()) std::printf("PROBE end workers=%d\n", pool.active_workers());
        }
    } else if (line == "PROBE adapt" || line.rfind("PROBE adapt ", 0) == 0) {
        {
            if (ba_job) { ba_thr.join(); ba_job = false; }
            (void) probe_adapt(line, perr);
        }
    } else if (line != "PROBE rates") perr = "PROBE: expected PROBE rates, PROBE set or PROBE adapt";
    else {
        if (ba_job) { ba_thr.join(); ba_job = false; }
        basalt::program::ProbeEnv penv;
        penv.pool = &pool;
        penv.ram_blob = [this](int64_t l, int64_t e) { return srcp->resident_blob(l, e); };
        if (const basalt::kernels::cpu::ExpertLayout& lay = basalt::kernels::cpu::expert_layout(); lay.native) penv.fmt = lay.fmt;
        penv.peer_device = peer.valid() ? peer.device() : -1;
        penv.peer_path = basalt::models::qwen4exp::make_probe_peer_path();
        penv.n_layers = g.n_layers;
        penv.n_expert = g.n_expert;
        basalt::program::probe_rates(penv, stdout, perr);
    }
}

void QfnRuntime::sleep_command(bool down, std::string& verr) {
    if (!service_vision(true)) { verr = err; std::exit(1); }
    // A peer card's tier goes too.  SLEEP then drops the RAM copy's registration with the driver (a process
    // checkpoint would map the copy into every context again on restore); WAKE registers it again while
    // both caches refill from the ranges already back.  A tier or a registration that cannot come back
    // fails WAKE: the server then starts the engine again.
    std::string perr, reply, uerr;
    const auto tp = Clock::now();
    const bool repin = !down && src.complement_unpinned();
    std::function<bool(const void*, uint64_t)> ready;
    if (repin) {
        src.repin_start(4);
        ready = [this](const void* p, uint64_t n) { return src.repin_wait(p, n); };
    }
    const bool peer_ok = !peer.valid() || (down ? peer.sleep(perr) : peer.wake(perr, ready));
    vram_command(down ? "VRAM " + std::to_string(1ll << 30) : std::string("VRAM"), verr, true, &reply);
    if (repin) {
        if (!src.repin_finish(uerr) && verr.empty()) verr = "WAKE: " + uerr;
        basalt::log::info("basalt serve", "WAKE: the RAM copy registered again beside the caches' refill "
                          "in %.0f ms", std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
    }
    if (down && verr.empty() && src.complement_ready()) {
        const auto tu = Clock::now();
        if (src.unpin_complement(uerr))
            basalt::log::info("basalt serve", "SLEEP: the RAM copy's registration dropped in %.0f ms",
                              std::chrono::duration<double, std::milli>(Clock::now() - tu).count());
        else
            basalt::log::info("basalt serve", "SLEEP: the RAM copy stays registered (%s)", uerr.c_str());
    }
    if (!down && peer.valid() && peer_ok && !peer.wake_wait(perr)) {
        if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
    } else if (!down && !peer_ok) {
        if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
    } else if (!peer_ok) {
        basalt::log::warn("basalt serve", "SLEEP: the peer tier stays as it is (%s)", perr.c_str());
    } else if (peer.valid()) {
        basalt::log::info("basalt serve", "%s: the peer tier %s; %.0f ms with the main cache",
                          line.c_str(), down ? "gave its VRAM back" : "is back",
                          std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
    }
    if (verr.empty()) std::fputs(reply.c_str(), stdout);
}

void QfnRuntime::between_requests(const std::string& line) {
    if (!service_vision(true)) {
        basalt::log::error("basalt serve", "peer vision lease: %s", err.c_str());
        std::exit(1);
    }
    // every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
    if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
        Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
        save_profile("periodic");
    if (ba_job) { ba_thr.join(); ba_job = false; }   // (a request runs the same copy steps itself)
    if (line != "QUIT" && !batch_on() && !vram_evicted.empty() &&
        shared_pages && basalt::core::exp_flag("shared_kv_expert_lending")) {
        std::string error;
        if (!kv_drain(error) || !cache_regrow(error))
            basalt::log::warn("basalt serve", "shared KV expert refill: %s", error.c_str());
    }
}

void QfnRuntime::after_input_start() { sp.should_stop = [&] { return halted(); }; }

int QfnRuntime::scheduler_pressure_class(bool require_fresh) {
    if (!scheduler_enabled) return require_fresh ? -1 : 0;
    const auto now = Clock::now();
    if (require_fresh || now - scheduler_checked >= std::chrono::milliseconds(200)) {
        std::lock_guard<std::mutex> lock(in_mu);
        scheduler_pressure = scheduler_queue;
        scheduler_checked = now;
    }
    if (now - scheduler_pressure.received > std::chrono::seconds(4)) return require_fresh ? -1 : 0;
    if (scheduler_pressure.parked > 0) return 3;
    if (scheduler_pressure.control > 0 || scheduler_pressure.admission > 0)
        return scheduler_pressure.oldest_ms >= 1000 ? 2 : 1;
    return 0;
}

void QfnRuntime::observe_scheduler(SchedulerPhase phase, int64_t prompt_left) {
    if (!scheduler_enabled) return;
    const auto now = Clock::now();
    if (prompt_left >= 0) scheduler_prompt_left = prompt_left;
    const bool changed = phase != scheduler_phase;
    if (changed) {
        scheduler_phase = phase;
        ++scheduler_epoch;
        if (phase != SchedulerPhase::batch_decode) {
            decode_rows.reset();
            pcie_state = "idle";
            pcie_applied_q8 = 0;
        }
        if (phase != SchedulerPhase::overlap) scheduler_quantum_ms = 0;
        if (phase == SchedulerPhase::idle || phase == SchedulerPhase::solo_decode ||
            phase == SchedulerPhase::batch_decode)
            scheduler_prompt_left = scheduler_chunk = 0;
    }
    if (!changed && now - scheduler_reported < std::chrono::milliseconds(500)) return;
    scheduler_reported = now;
    const auto pressure = scheduler_pressure_class();
    const bool fresh = now - scheduler_pressure.received <= std::chrono::seconds(4);
    const char* phase_names[] = {"idle", "prefill", "overlap", "solo_decode", "batch_decode"};
    const auto* primary = pf_parts.empty() ? nullptr : &pf_parts.front();
    const int64_t prefill_loan = primary && primary->cache == &xcache && primary->first_now >= 0
        ? xcache.bytes() - xcache.bytes_of(primary->first_now) : 0;
    uint64_t index_mapped = 0, peer_prefill_host = 0, peer_prefill_scratch = 0;
    std::set<const basalt::core::KvArena*> seen;
    for (const auto& device_slots : bslot_ss)
        for (const auto& slot : device_slots)
            for (int64_t q = 0; slot && slot->qsa_states && q < slot->qsa_alloc; ++q)
                if (const auto* arena = slot->qsa_states[slot->qsa_ord0 + q].shared_index_arena;
                    arena && seen.insert(arena).second)
                    index_mapped += arena->mapped_bytes();
    for (const auto& stage : pstages) {
        if (stage->image) peer_prefill_host += stage->image_bytes;
        if (stage->lent_slots > 0 && stage->buf) peer_prefill_scratch += stage->buf_bytes;
    }
    uint64_t recovery_bytes = 0, disk_bytes = 0, recovery_queued = 0;
    uint64_t parked_bytes = conversations ? conversations->bytes() : 0;
    for (const auto& cache : parked_conversations)
        if (cache) parked_bytes += cache->bytes();
    for (unsigned mode = 0; mode < cache_count; ++mode) {
        recovery_bytes += recovery_stores[mode].stats().bytes;
        recovery_queued += recovery_stores[mode].queued_bytes();
        disk_bytes += disk_store(mode).stats().bytes;
    }
    const auto expert_arena_bytes = o.mmap_experts ? src.resident_bytes()
                                                   : basalt::kernels::cpu::expert_layout().total;
    std::printf("SCHEDSTAT phase=%s epoch=%llu prompt_remaining=%lld chunk_tokens=%lld decode_quantum_ms=%.0f "
                "verify_rows=%d verify_limit=%d verify_trial_rows=%d verify_trial_remembered=%d "
                "decision=%s batch_output_gap_ms=%.0f "
                "batch_windows=%llu limited_batch_windows=%llu batch_window_ms=%.0f limited_batch_window_ms=%.0f "
                "queue_fresh=%d pressure=%d control_waiters=%d admission_waiters=%d parked_waiters=%d "
                "oldest_wait_ms=%lld free_kv_pages=%lld "
                "gpu0_expert_slots=%lld gpu0_expert_cache_bytes=%lld gpu0_expert_capacity_bytes=%lld "
                "gpu0_expert_mapped_bytes=%lld gpu0_prefill_loan_bytes=%lld peer_resident_experts=%lld "
                "peer_expert_cache_bytes=%lld peer_vision_loan_bytes=%lld mtp_tracked_bytes=%lld "
                "gpu0_cache_hits=%lld peer_expert_entries=%lld cpu_expert_entries=%lld pcie_distinct_experts=%lld "
                "peer_swaps=%lld expert_arena_bytes=%lld vision_lease=%d "
                "pinned_kv_bytes=%llu shared_index_mapped_bytes=%llu pinned_peer_weights_bytes=%llu "
                "peer_prefill_scratch_within_loan_bytes=%llu recovery_cache_bytes=%llu disk_cache_bytes=%llu "
                "cuda0_prefill_workspace_bytes=%llu cuda0_prefill_workspace_borrowed=%d "
                "vision_device_bytes=%llu ram_prefix_cache_total_bytes=%llu "
                "recovery_queued_bytes=%llu capture_ram_available_bytes=%lld "
                "pcie_tuner_state=%s pcie_frac_q8=%d pcie_trial_q8=%d pcie_tuner_windows=%llu "
                "pcie_stage_payload_bytes=%llu pcie_window_eligible=%d pcie_window_staged=%d pcie_window_payload_mbps=%lld\n",
                phase_names[(int) phase], (unsigned long long) scheduler_epoch,
                (long long) scheduler_prompt_left, (long long) scheduler_chunk, scheduler_quantum_ms,
                scheduler_rows, scheduler_hard_rows, scheduler_trial_rows, (int) decode_rows.trial_from_profile(),
                scheduler_decision,
                scheduler_output_gap_ms,
                (unsigned long long) scheduler_batch_windows, (unsigned long long) scheduler_limited_windows,
                scheduler_batch_ms, scheduler_limited_ms,
                (int) fresh, pressure, fresh ? scheduler_pressure.control : -1,
                fresh ? scheduler_pressure.admission : -1, fresh ? scheduler_pressure.parked : -1,
                fresh ? scheduler_pressure.oldest_ms : -1, fresh ? scheduler_pressure.free_pages : -1,
                (long long) xcache.slots(), (long long) xcache.bytes(), (long long) xcache.full_bytes(),
                (long long) xcache.mapped_bytes(), (long long) prefill_loan,
                (long long) (peer.valid() ? peer.resident() : 0),
                (long long) (peer.valid() ? peer.gib() * 1073741824.0 : 0.0),
                (long long) (vision_decode_lease.load() ? vision_tail_bytes : 0),
                (long long) (mtp_on ? mtp.vram_bytes() : 0),
                (long long) drive.d.cache_hits, (long long) drive.d.peer_entries,
                (long long) drive.d.multi_entries, (long long) drive.d.pcie_experts,
                (long long) (peer.valid() ? peer.swaps() : 0), (long long) expert_arena_bytes,
                (int) vision_decode_lease.load(),
                (unsigned long long) basalt::core::qsa_kv_host_bytes(), (unsigned long long) index_mapped,
                (unsigned long long) peer_prefill_host, (unsigned long long) peer_prefill_scratch,
                (unsigned long long) recovery_bytes, (unsigned long long) disk_bytes,
                (unsigned long long) sp.workspace_bytes(), (int) sp.workspace_borrowed(),
                (unsigned long long) vis_enc.device_bytes(), (unsigned long long) parked_bytes,
                (unsigned long long) recovery_queued,
                (long long) basalt::core::conversation_available_memory().value_or(-1),
                ver.batch_pcie_tunable() ? pcie_state : "unavailable",
                ver.batch_pcie_tunable() ? pcie_applied_q8 : -1, -1,
                (unsigned long long) pcie_tuner_windows, (unsigned long long) pcie_tuner_bytes,
                pcie_last_window.eligible, pcie_last_window.staged, pcie_window_payload_mbps);
    std::fflush(stdout);
}

void QfnRuntime::print_info() {
    if (scheduler_enabled) {
        std::printf("INFO scheduler_queue=1 dense_arena_cuda0_bytes=%llu native_head_cuda0_bytes=%llu "
                    "adapter_cuda0_payload_bytes=%llu adapter_peer_payload_bytes=%llu disk_cache_budget_bytes=%llu "
                    "recovery_budget_bytes=%llu verify_arena_cuda0_bytes=%llu\n",
                    (unsigned long long) wt.report().arena_bytes, (unsigned long long) native_head.weight_bytes(),
                    (unsigned long long) (lora_main && lora_host ? lora_host->factor_bytes() : 0),
                    (unsigned long long) (lora_peer && lora_host ? lora_host->factor_bytes() : 0),
                    (unsigned long long) o.conversation_cache_disk_mib << 20,
                    recovery_stores[0].enabled() ? (unsigned long long) o.conversation_cache_mib << 20 : 0,
                    (unsigned long long) (ver.arena_bytes() + ver_b.arena_bytes()));
        for (const auto& stage : stages)
            std::printf("INFO dense_arena_cuda%d_bytes=%llu native_head_cuda%d_bytes=%llu verify_arena_cuda%d_bytes=%llu\n",
                        stage->dev, (unsigned long long) stage->wt.report().arena_bytes,
                        stage->dev, (unsigned long long) stage->head.weight_bytes(),
                        stage->dev, (unsigned long long) (stage->ver.arena_bytes() + stage->ver_b.arena_bytes()));
    }
    if (recovery_stores[0].enabled()) std::printf("INFO recovery_ram=1\n");
    if (vision_overlap) std::printf("INFO vision_overlap=1 vision_lazy=1\n");
    // what the server's Monitor tab shows (the server skips unknown lines until READY)
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // "Experts in VRAM" is every stage's cache, not one card's.  The caches hold disjoint pairs, so they add.
        const int64_t slots_primary = (int64_t) xcache.slots();
        const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
        int64_t slots_all = slots_primary, mib_all = mib_primary;
        for (const auto& st : stages) {
            slots_all += (int64_t) st->cache.slots();
            mib_all += (int64_t) (st->cache.bytes() >> 20);
        }
        // the pack's quantization label (the last word of its name, without the publisher's prefixes) and the PCI
        // bus ids of the cards this engine computes on, primary first: what a benchmark row records
        std::string quant_label, gpu_ids;
        {
            const basalt::core::Json* nm = model_file.meta().get("name");
            quant_label = basalt::program::pack_quant_label(nm != nullptr && nm->is_string() ? nm->s : std::string());
            for (int d : basalt::program::engine_gpu_order(o.peer_device, o.peer_devices,
                                                           multi_gpu ? split_devs : std::vector<int>())) {
                char bus[32] = {};
                if (cudaDeviceGetPCIBusId(bus, sizeof bus, d) == cudaSuccess)
                    gpu_ids += (gpu_ids.empty() ? "" : ",") + std::string(bus);
            }
        }
        std::printf("INFO logit_bias=%lld context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                    "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                    "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                    "conversation_cache_mib=%lld conversation_cache_slots=%d "
                    "conversation_cache_min_free_mib=%lld conversation_cache_min_tokens=%d tail_role_token=%lld vram_elastic=%d%s quant=%s gpus=%s engine=" BASALT_VERSION "\n",
                    (long long) n_vocab, (long long) o.max_context, o.kv.c_str(),
                    (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                     ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                    (long long) slots_all, (long long) mib_all,
                    (long long) slots_primary, (long long) mib_primary,
                    o.spec, o.mtp_max_t,
                    o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                    (long long) ((o.mmap_experts ? src.resident_bytes() : basalt::kernels::cpu::expert_layout().total) >> 20),
                    pool.workers(), o.pcie_frac,
                    (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                    (long long) o.conversation_cache_min_free_mib, o.conversation_cache_min_tokens, (long long) o.tail_role_token, o.vram_elastic && xcache.segmented() ? 1 : 0,
                    o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                   " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0) +
                                   " slot0_solo=" + std::to_string(slot0_is_solo ? 1 : 0) +
                                   " solo_any_slot=" + std::to_string(slot0_is_solo ? 1 : 0)).c_str() : "",
                    quant_label.c_str(), gpu_ids.c_str());
        if (shared_pages)
            std::printf("INFO shared_kv_pages=%u shared_kv_page_cells=%lld shared_kv_mixed_adapters=%d shared_kv_grow=1\n", shared_pages->capacity(),
                        (long long) basalt::kernels::qsa_real_shapes().page_size, configured_adapters() ? 1 : 0);
        if (main_link_gbps > 0.0)
            std::printf("INFO pcie_h2d_cuda0_mbps=%lld\n", (long long) (main_link_gbps * 1000.0 + 0.5));
        for (const auto& stage : pstages)
            if (stage->link_gbps > 0.0)
                std::printf("INFO pcie_h2d_cuda%d_mbps=%lld\n", stage->dev,
                            (long long) (stage->link_gbps * 1000.0 + 0.5));
        if (rope_context_profiles) std::printf("INFO rope_context_profiles=1\n");
        if (shared_pages && slot0_solo() && rope_context_profiles && !pipe_on && !piped && stages.empty() &&
            conversations->enabled() && recovery_stores[0].enabled() && basalt::core::exp_flag("cold_shared_recovery"))
            std::printf("INFO cold_shared_recovery_min_tokens=%lld\n",
                        (long long) std::max<int64_t>(disk_min_tokens(), o.conversation_cache_min_tokens));
        if (shared_pages && basalt::core::exp_flag("grouped_short_prefill") && o.short_read > 1)
            std::printf("INFO grouped_short_prefill=%lld\n", (long long) o.short_read);
        // whether the in-engine encoder reads video (FFmpeg found at run time), for the server's start line
        if (o.vision_in_engine) std::printf("%s\n", basalt::vision::video_line().c_str());
    }
}

Flow QfnRuntime::run_request() {
    QfnRequest request(*this);
    return request.execute();
}

void QfnRuntime::bind_lora_ple(basalt::core::SessionState& session,
                               const basalt::qwen4exp::AtomicLoraDevice* device) {
    if (!device) return;
    session.ple.w.value_lora_a = device->input_factor("blk.1.ple_value.weight");
    session.ple.w.value_lora_b = device->output_factor();
    session.ple.w.value_lora_scale = device->scale();
}

int QfnRuntime::init_atomic_lora() {
    if (o.atomic_lora.empty()) return 0;
    auto* embed = basalt::core::native_embed();
    if (!embed || (o.batch > 1 && !shared_pages) || !split_devs.empty() || !xpeers.empty() || !o.dflash2.empty() ||
        g.n_layers != 48 || g.n_embd != 2560 || g.n_ff != 640 || g.n_expert != 512) {
        basalt::log::error("basalt serve", "AtomicChat LoRA requires native Flash-Next in solo or shared KV mode");
        return 1;
    }
    try {
        std::vector<basalt::qwen4exp::AtomicLoraSite> sites;
        auto add = [&](const std::string& name, bool embedding = false) {
            const auto* weight = wt.find(name);
            if (!weight || weight->ne0 <= 0 || weight->ne1 <= 0)
                throw std::runtime_error("missing base writer " + name);
            sites.push_back({name, (size_t) weight->ne0, (size_t) weight->ne1, 1, embedding});
        };
        add("token_embd.weight", true);
        add("blk.1.ple_value.weight");
        for (int l = 0; l < g.n_layers; ++l) {
            const std::string layer = "blk." + std::to_string(l) + ".";
            add(layer + (basalt::core::is_qsa_layer(g, l) ? "attn_output.weight" : "ssm_out.weight"));
            add(layer + "ffn_down_shexp.weight");
            sites.push_back({layer + "ffn_down_exps.weight", (size_t) g.n_ff,
                             (size_t) g.n_embd, (size_t) g.n_expert});
        }
        lora_host = std::make_unique<basalt::qwen4exp::AtomicLora>(o.atomic_lora, sites);
        lora_main = std::make_unique<basalt::qwen4exp::AtomicLoraDevice>(*lora_host, 0);
        if (peer.valid()) lora_peer = std::make_unique<basalt::qwen4exp::AtomicLoraDevice>(*lora_host, peer.device());
        embed->bind_lora(0, *lora_main);
        if (lora_peer) embed->bind_lora(peer.device(), *lora_peer);
        bind_lora_ple(ss, lora_main.get());
        if (shared_pages)
            for (auto& slot : bslot_ss[0]) bind_lora_ple(*slot, lora_main.get());
        basalt::log::info("basalt serve", "AtomicChat LoRA factors resident at zero scale (%zu bytes)",
                          lora_host->factor_bytes());
    } catch (const std::exception& error) {
        basalt::log::error("basalt serve", "AtomicChat LoRA: %s", error.what());
        return 1;
    }
    return 0;
}

void QfnRuntime::report_overlap_learning(const char* outcome, uint64_t scope, int64_t chunk) {
    std::printf("SCHEDLEARN %s %llu %016llx %lld\n", outcome, (unsigned long long) ++overlap_episode,
                (unsigned long long) scope, (long long) chunk);
    std::fflush(stdout);
    basalt::log::info("basalt batch", "overlap learning %s: scope %016llx, chunk %lld", outcome,
                      (unsigned long long) scope, (long long) chunk);
}

void QfnRuntime::reject_overlap_learning(uint64_t scope, int64_t chunk) {
    overlap_learned.erase(scope);
    if (overlap_rejected.size() >= 16) overlap_rejected.erase(overlap_rejected.begin());
    overlap_rejected[scope] = chunk;
    report_overlap_learning("drop", scope, chunk);
}

void QfnRuntime::observe_overlap_emission(size_t slot, Clock::time_point emitted) {
    auto& sample = *overlap_settlement;
    if (scheduler_pressure_class(true) != 0 || async_image_active() ||
        std::any_of(sample.decoders.begin(), sample.decoders.end(), [&](const auto& decoder) {
            if (decoder.slot >= bs.size()) return true;
            const auto& state = bs[decoder.slot];
            const auto& sequence = *shared_sequences[decoder.slot];
            return !state.active || state.stop || state.t0 != decoder.started || state.p / 32768 != decoder.context ||
                   cached_context_key((int) decoder.slot) != decoder.cache_key || !sequence.image_rope.empty();
        })) {
        overlap_settlement.reset();
        return;
    }
    for (auto& decoder : sample.decoders) {
        if (decoder.slot != slot || decoder.seen) continue;
        const double gap = std::chrono::duration<double, std::milli>(emitted - decoder.last_emit).count();
        basalt::log::info("basalt batch", "overlap learning settlement: slot %zu gap %.1f ms, limit %.1f ms",
                          slot, gap, sample.gap_limit_ms);
        if (gap > sample.gap_limit_ms) {
            reject_overlap_learning(sample.scope, sample.chunk);
            overlap_settlement.reset();
            return;
        }
        decoder.seen = true;
    }
    if (std::all_of(sample.decoders.begin(), sample.decoders.end(), [](const auto& decoder) { return decoder.seen; })) {
        report_overlap_learning("win", sample.scope, sample.chunk);
        overlap_settlement.reset();
    }
}

int QfnRuntime::init_before_input() {
    const auto take = [](std::string_view& fields, auto& value, int base) {
        const auto comma = fields.find(',');
        if (comma != std::string_view::npos && comma == fields.size() - 1) return false;
        const auto field = fields.substr(0, comma);
        if (field.empty()) return false;
        const auto end = field.data() + field.size();
        const auto result = std::from_chars(field.data(), end, value, base);
        fields = comma == std::string_view::npos ? std::string_view{} : fields.substr(comma + 1);
        return result.ec == std::errc{} && result.ptr == end;
    };
    if (const char* learned = std::getenv("BASALT_NATIVE_OVERLAP_LEARNING"); scheduler_enabled && learned) {
        std::string_view entries(learned);
        overlap_learning_enabled = entries.size() <= 1024;
        while (overlap_learning_enabled && !entries.empty() && overlap_learned.size() < 16) {
            const auto semi = entries.find(';');
            auto fields = entries.substr(0, semi);
            entries = semi == std::string_view::npos ? std::string_view{} : entries.substr(semi + 1);
            uint64_t scope = 0;
            int64_t chunk = 0, expires = 0;
            if (take(fields, scope, 16) && take(fields, chunk, 10) && take(fields, expires, 10) &&
                fields.empty() && scope && chunk > 0 && chunk <= 32768 && chunk % 256 == 0 && expires > 0)
                overlap_learned.emplace(scope, std::make_pair(chunk, expires));
        }
    }
    if (const char* hint = std::getenv("BASALT_NATIVE_OVERLAP_HINT"); scheduler_enabled && hint) {
        std::string_view fields(hint);
        uint64_t scope = 0;
        if (take(fields, scope, 16) && take(fields, overlap_hint_chunk, 10) && take(fields, overlap_hint_gap_ms, 10) &&
            take(fields, overlap_hint_expires, 10) && fields.empty() && overlap_hint_chunk > 0 &&
            overlap_hint_chunk % 256 == 0 && overlap_hint_gap_ms >= 0 && overlap_hint_gap_ms <= 60000)
            overlap_hint_scope = scope;
    }
    if (!shared_tables.empty()) {
        const auto capacity = static_cast<uint32_t>(ss.qsa_states[ss.qsa_primary()].host_pool_pages);
        shared_pages = std::make_unique<basalt::core::SharedKvPages>(capacity);
        for (size_t slot = 0; slot < shared_tables.size(); ++slot) {
            auto sequence = std::make_unique<SharedSequence>();
            sequence->stages.push_back({0, *shared_tables[slot], *bslot_ss[0][slot], nullptr});
            shared_sequences.push_back(std::move(sequence));
        }
    }
    rope_context_profiles = basalt::core::exp_flag("rope_context_profiles") != 0;
    if (rope_context_profiles && (!shared_pages || o.max_context < 524288)) {
        basalt::log::error("basalt serve", "RoPE context profiles require a shared KV pool with at least 524288 cells");
        return 1;
    }
    if (const int code = init_atomic_lora()) return code;
    if (swift_main && o.batch > 1 && !shared_pages) {
        basalt::log::error("basalt serve", "Swift LoRA batch serving requires shared KV");
        return 1;
    }
    if (const int code = init_prompt()) return code;
    if (lora_main) sp.bind_lora(lora_main.get(), lora_peer.get());
    if (const int code = init_verify()) return code;
    if (const int code = init_conversations()) return code;
    return init_tiers();
}

int QfnRuntime::init_after_input() {
    using basalt::kernels::RopeContext;
    const basalt::kernels::RopeContextScope scope(rope_context_profiles ? RopeContext::Native : RopeContext::Startup);
    if (const int code = init_memory()) return code;
    if (const int code = capture_windows()) return code;
    return init_batch();
}

void QfnRuntime::at_exit() {
    for (int st = 0; st < std::max(n_stages, 1); ++st)
        for (const std::string& r : stage_ver(st).route_report()) basalt::log::info("basalt serve", "%s", r.c_str());
    save_profile("exit");   // QUIT, or the server closed stdin
}

int run(Context& context) {
    QfnRuntime runtime(context);
    runtime.pre_ = {"qwen35", basalt::models::qwen4exp::tokenizer::pretok_qwen35};
    return runtime.execute();
}

}  // namespace basalt::models::qwen4exp::serve
