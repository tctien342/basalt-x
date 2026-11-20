// src/models/qwen4exp/serve/prompt_read.cpp - request prompt_read operations.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "prefill_chunk.hpp"
#include "overlap_trial.hpp"
#include <ctime>
#include <tuple>

namespace basalt::models::qwen4exp::serve {

int64_t QfnRequest::read_alone_limit() const {
    return runtime_.shared_pages && runtime_.batch_on() && basalt::core::exp_flag("shared_kv_live_prefill")
        ? 1024 : runtime_.kReadAlone;
}

bool QfnRequest::read_part(int64_t a0, int64_t b0, std::string& e, int64_t* solo_at) {
    // (a layer split reads its stages as a pipeline over one run's chunks: in pieces only beside slots)
    if (runtime_.o.batch <= 0 || runtime_.piped || (!runtime_.stages.empty() && !runtime_.batch_on()) || (b0 - a0 <= read_alone_limit() && resumed_chunk <= 0)) {
        // no window runs meanwhile, so the CPU pool is free: it takes its share of the streamed experts
        const bool lend_pool = runtime_.o.batch > 0 && !runtime_.multi_gpu && !runtime_.o.no_pool;
        if (lend_pool) { runtime_.sp.set_cpu_pool(&runtime_.pool); runtime_.sp.set_cpu_share(runtime_.o.prefill_cpu_share); }
        const bool ok = runtime_.sp.run(ids.data() + a0, b0 - a0, a0, e);
        if (lend_pool) runtime_.sp.set_cpu_pool(nullptr);
        return ok;
    }
    // Shared residency maps are mutated by both prefill and decode. Alternate at chunk boundaries.
    if (runtime_.batch_on() && runtime_.stages.empty() && !runtime_.shared_pages) {
        // the whole range is read on a side thread, in the prompt path's own chunks, while this thread keeps
        // running the slots' windows; the work that touches the draft layer or the output runs here, between
        // two windows, when the prompt path asks for it
        if (!runtime_.xq_drain()) { e = "an adaptive refill failed"; return false; }
        runtime_.apply_pending(true);
        struct Mail {
            std::mutex mu;
            std::condition_variable cv;
            std::function<void()> job;
            bool job_done = false, fin = false;
        } mail;
        std::atomic<bool> abort_read{false};
        bool ok = false;
        std::string terr;
        int yield_req = -1;                 // a BYIELD <slot> not acted on yet
        int64_t yield_q = -1;               // the chunk boundary the read stopped at, to give way
        int yield_slot = -1;
        auto orig_chunk = runtime_.sp.on_chunk;
        auto orig_stop = runtime_.sp.should_stop;
        runtime_.sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& ce) -> bool {
            bool r = false;
            std::unique_lock<std::mutex> lk(mail.mu);
            mail.job = [&] {
                r = orig_chunk ? orig_chunk(R_rows, T, p0, ce) : true;
                if (!r || yield_req < 0) return;
                // a shorter request waits: the read stops after this chunk (the draft layer has it) when the
                // slot can take what was read, or goes on
                const int ys = yield_req;
                yield_req = -1;
                if (can_yield(p0 + T, b0, ys, true)) {
                    yield_q = p0 + T;
                    yield_slot = ys;
                    abort_read.store(true);
                } else {
                    basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld", ys,
                                 (long long) (p0 + T), (long long) n);
                }
            };
            mail.job_done = false;
            mail.cv.wait(lk, [&] { return mail.job_done; });
            return r;
        };
        runtime_.sp.should_stop = [&] { return abort_read.load() || (orig_stop && orig_stop()); };
        runtime_.ov_active = true;
        std::thread th([&, context = basalt::kernels::rope_context()] {
            const basalt::kernels::RopeContextScope scope(context);
            const basalt::core::OnDevice on(0);
            ok = runtime_.sp.run(ids.data() + a0, b0 - a0, a0, terr);
            if (ok && runtime_.ov_post) {   // (the checkpoint's copy to the host beside the windows)
                ok = runtime_.ov_post();
                runtime_.ov_post_done = ok;
                if (!ok) terr = "saving a conversation checkpoint failed" + runtime_.ckpt_why;
            }
            std::lock_guard<std::mutex> lk(mail.mu);
            mail.fin = true;
            mail.cv.notify_all();
        });
        bool fatal = false;
        int64_t ov_jobs = 0, ov_wins = 0;
        double ov_job_ms = 0, ov_job_max = 0, ov_win_ms = 0, ov_win_max = 0;
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mail.mu);
                if (mail.job && !mail.job_done) job = mail.job;
                else if (mail.fin) break;
            }
            if (job) {
                const auto tj = Clock::now();
                job();
                const double jm = std::chrono::duration<double, std::milli>(Clock::now() - tj).count();
                ++ov_jobs; ov_job_ms += jm; ov_job_max = std::max(ov_job_max, jm);
                std::lock_guard<std::mutex> lk(mail.mu);
                mail.job = nullptr;
                mail.job_done = true;
                mail.cv.notify_all();
                continue;
            }
            {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                std::lock_guard<std::mutex> lk(runtime_.in_mu);
                for (auto it = runtime_.in_lines.begin(); it != runtime_.in_lines.end();) {
                    if (it->rfind("BSTOP ", 0) == 0) {
                        const int b = std::atoi(it->c_str() + 6);
                        if (b >= 0 && b < (int) runtime_.bs.size()) runtime_.bs[(size_t) b].stop = true;
                        it = runtime_.in_lines.erase(it);
                    } else if (it->rfind("BYIELD ", 0) == 0) {
                        yield_req = std::atoi(it->c_str() + 7);
                        it = runtime_.in_lines.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            if (!fatal && runtime_.batch_on()) {
                const auto tw = Clock::now();
                if (!runtime_.batch_step()) { fatal = true; abort_read.store(true); }
                const double wm = std::chrono::duration<double, std::milli>(Clock::now() - tw).count();
                ++ov_wins; ov_win_ms += wm; ov_win_max = std::max(ov_win_max, wm);
            } else {
                std::unique_lock<std::mutex> lk(mail.mu);
                mail.cv.wait_for(lk, std::chrono::milliseconds(2), [&] { return mail.fin || (mail.job && !mail.job_done); });
            }
        }
        th.join();
        runtime_.ov_active = false;
        basalt::log::info("basalt batch", "overlapped prompt read: %lld windows (%.1f ms avg, %.0f ms max) beside "
                             "%lld chunk jobs run between windows (%.1f ms avg, %.0f ms max)",
                     (long long) ov_wins, ov_wins ? ov_win_ms / (double) ov_wins : 0.0, ov_win_max,
                     (long long) ov_jobs, ov_jobs ? ov_job_ms / (double) ov_jobs : 0.0, ov_job_max);
        runtime_.sp.on_chunk = std::move(orig_chunk);
        runtime_.sp.should_stop = std::move(orig_stop);
        if (yield_q >= 0 && !fatal) {   // the read stopped at a chunk boundary to give way (BYIELD)
            std::string ye;
            const auto ty = Clock::now();
            if (yield_q >= 2) {   // a run that ends early leaves the token window of the PLE hash where it began
                runtime_.rsp->ple_prev[0] = (int32_t) ids[(size_t) yield_q - 2];
                runtime_.rsp->ple_prev[1] = (int32_t) ids[(size_t) yield_q - 1];
            }
            if (!park_read(yield_q, yield_slot, ye, runtime_.sp.chunk())) {
                e = "BYIELD: parking the prompt read failed: " + ye;
                return false;
            }
            basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                 "holds it (copied in %.1f ms)", (long long) yield_q, (long long) n, yield_slot,
                         std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
            e = "yield";
            return false;
        }
        if (fatal) { batch_fatal = true; e = "a batch window failed"; return false; }
        if (!ok) e = terr;
        return ok;
    }
    const int64_t C = std::max<int64_t>(runtime_.sp.chunk(), 1);
    constexpr int64_t fixed_chunk = 1024;
    const bool adaptive = runtime_.o.prefill_auto_fit && read_alone_limit() == fixed_chunk;
    const bool adaptive_quantum = adaptive &&
        (runtime_.scheduler_enabled || basalt::core::exp_flag("adaptive_prefill_quantum"));
    const int64_t layout_cap = std::min(C, kInterleavedPrefillCap);
    InterleavedPrefillChunk policy(layout_cap);
    const double chunk_override = basalt::core::exp_flag("tuner_overlap_chunk");
    const double share_override = basalt::core::exp_flag("tuner_overlap_decode_pct");
    const auto measured_chunk = adaptive && chunk_override > 0.0
        ? std::min<int64_t>(std::max<int64_t>(256, (int64_t) std::min<double>(chunk_override, layout_cap) / 256 * 256),
                            layout_cap) : 0;
    const auto measured_share = adaptive && share_override > 0.0
        ? std::clamp(share_override, 25.0, 100.0) / 100.0 : 0.0;
    const bool measure_overlap = measured_chunk > 0 || measured_share > 0.0;
    using Trial = OverlapTrial<basalt::kernels::kVerifyMaxT>;
    using Decoder = OverlapDecoder;
    const auto cohort = [&] {
        std::vector<Decoder> active;
        for (size_t b = 0; b < runtime_.bs.size(); ++b) {
            const auto& slot = runtime_.bs[b];
            if (slot.active) active.emplace_back(b, slot.t0, slot.p / 32768, runtime_.cached_context_key((int) b));
        }
        return active;
    };
    const double manual_trial = basalt::core::exp_flag("tuner_overlap_trial_chunk");
    const bool hinted = manual_trial == 0.0 && runtime_.overlap_hint_scope &&
                        std::time(nullptr) < runtime_.overlap_hint_expires;
    const bool learning = runtime_.overlap_learning_enabled && manual_trial == 0.0 && !hinted &&
                          !overlap_learning_attempted;
    const double trial_option = hinted ? runtime_.overlap_hint_chunk : manual_trial;
    // Qualified legacy hints own their explicit budget; autonomous observations always use the relative policy.
    const double gap_budget_ms = hinted ? runtime_.overlap_hint_gap_ms
        : learning ? 0.0 : basalt::core::exp_flag("tuner_overlap_gap_ms");
    const auto text_overlap = [&](const std::vector<Decoder>& decoders) {
        if (geni || !emb_path.empty() || !runtime_.mrope_identity || runtime_.async_image_active()) return false;
        return std::none_of(decoders.begin(), decoders.end(), [&](const Decoder& decoder) {
            return !runtime_.shared_sequences[std::get<0>(decoder)]->image_rope.empty();
        });
    };
    const auto scope_for = [&](const std::vector<Decoder>& decoders) {
        // Width, request adapter and work shape precede the decoder multiset. Slot identity stays process-local.
        uint64_t scope = 14695981039346656037ull;
        const auto field = [&](uint64_t v) { scope = (scope ^ v) * 1099511628211ull; };
        field(layout_cap); field(request_adapter()); field(n / 32768); field(a0 / 32768); field((b0 - a0) / 16384);
        if (runtime_.rope_context_profiles) field(basalt::kernels::rope_context_index());
        std::vector<int64_t> decoder_shapes;
        const unsigned modes = runtime_.rope_context_profiles ? QfnRuntime::cache_count :
                               runtime_.swift_main ? QfnRuntime::adapter_count : 2;
        for (const auto& decoder : decoders)
            decoder_shapes.push_back(std::get<2>(decoder) * modes + std::get<3>(decoder));
        std::sort(decoder_shapes.begin(), decoder_shapes.end());
        field(decoder_shapes.size());
        for (const auto shape : decoder_shapes) field(shape);
        return scope;
    };
    int64_t candidate = trial_option >= 256.0 && trial_option <= layout_cap ? (int64_t) trial_option : 0;
    const bool trial_possible = adaptive && runtime_.shared_pages && !measure_overlap && decode_share > 0.0 &&
                                (learning ? layout_cap >= fixed_chunk + 256 :
                                    candidate > 0 && candidate == trial_option && candidate != fixed_chunk &&
                                    std::isfinite(gap_budget_ms) && gap_budget_ms >= 0);
    const int64_t trial_room = 6 * fixed_chunk + (learning ? 6 * (fixed_chunk + 256) : 2 * candidate);
    if (trial_option > 0.0 && b0 - a0 > trial_room)
        basalt::log::info("basalt batch", "overlap trial gate: requested %.0f, layout %lld, eligible %d, queue %d, remaining %lld, "
                          "adaptive %d, shared %d, override %d, gap budget %.1f ms",
                          trial_option, (long long) C, trial_possible, runtime_.scheduler_pressure_class(true),
                          (long long) (b0 - a0), adaptive, runtime_.shared_pages != nullptr, measure_overlap, gap_budget_ms);
    size_t slot_count = 0;
    bool trial_started = false, scope_miss_logged = false;
    uint64_t learning_scope = 0;
    bool learned_replay = false;
    std::vector<Decoder> trial_cohort;
    std::optional<Trial> trial;
    const auto changed = [&]() -> const char* {
        if (runtime_.bs.size() != slot_count || cohort() != trial_cohort) return "cohort_changed";
        if (runtime_.sp.chunk() != C) return "layout_changed";
        if ((hinted || learning) && !text_overlap(trial_cohort)) return "vision_activity";
        const int pressure = runtime_.scheduler_pressure_class(true);
        if (pressure < 0) return "queue_stale";
        if (pressure > 0) return "queue_busy";
        return nullptr;
    };
    const auto publish_trial = [&](const char* decision, const char* reason = "none") {
        runtime_.scheduler_decision = decision;
        basalt::log::info("basalt batch", "overlap trial: %s, candidate %lld, layout %lld, reason %s",
                          decision, (long long) candidate, (long long) C, reason);
    };
    int64_t measured_windows = 0, measured_slices = 0, measured_tokens = 0;
    double measured_budget_ms = 0.0, measured_decode_ms = 0.0;
    if (measure_overlap)
        basalt::log::info("basalt batch", "tuner overlap: requested %.0f tokens, layout %lld, fixed slice %lld, decode share %.2f",
                          chunk_override, (long long) C, (long long) measured_chunk, measured_share);
    struct DecodeProgress { int64_t produced; Clock::time_point when; };
    std::vector<DecodeProgress> progress;
    if (adaptive)
        for (const auto& slot : runtime_.bs)
            progress.push_back({slot.produced, runtime_.scheduler_enabled && slot.last_emit > slot.t0
                                               ? slot.last_emit : Clock::now()});
    for (int64_t q = a0; q < b0;) {
        if (trial_possible && !trial_started && b0 - q > trial_room && runtime_.scheduler_pressure_class(true) == 0) {
            auto active = cohort();
            const auto scope = scope_for(active);
            const bool text_only = text_overlap(active);
            if (learning) {
                learned_replay = false;
                candidate = fixed_chunk + 256;
                const auto known = runtime_.overlap_learned.find(scope);
                if (known != runtime_.overlap_learned.end() && known->second.first <= layout_cap &&
                    known->second.first > fixed_chunk && std::time(nullptr) < known->second.second) {
                    candidate = known->second.first;
                    learned_replay = true;
                }
                const auto failed = runtime_.overlap_rejected.find(scope);
                if (failed != runtime_.overlap_rejected.end() && failed->second == candidate)
                    candidate = 0;
                if (!text_only || !scope || candidate == 0 || b0 - q <= 6 * fixed_chunk + 6 * candidate)
                    candidate = 0;
            }
            const bool eligible = learning ? candidate != 0 : !hinted || (text_only && scope == runtime_.overlap_hint_scope);
            if (eligible) {
                trial_started = true;
                if (learning) {
                    learning_scope = scope;
                    overlap_learning_attempted = true;
                }
                slot_count = runtime_.bs.size();
                trial_cohort = std::move(active);
                trial.emplace(layout_cap, candidate, trial_cohort.size(), gap_budget_ms);
                if (trial->state() == Trial::State::interrupted) trial.reset();
                else {
                    basalt::log::info("basalt batch", "overlap trial scope: %016llx, source %s, text %d",
                                      (unsigned long long) scope,
                                      learned_replay ? "learned_replay" : learning ? "online_learning" :
                                      hinted ? "restart_hint" : "manual", text_only);
                    publish_trial("overlap_calibrating");
                }
            } else if (hinted && !scope_miss_logged) {
                scope_miss_logged = true;
                basalt::log::info("basalt batch", "overlap hint scope mismatch: %016llx, text %d",
                                  (unsigned long long) scope, text_only);
            }
        }
        if (trial) {
            if (const char* reason = changed()) {
                publish_trial("overlap_interrupted", reason);
                trial.reset();
            }
        }
        Trial::Tokens emitted{};
        if (trial)
            for (size_t i = 0; i < trial_cohort.size(); ++i)
                emitted[i] = -runtime_.bs[std::get<0>(trial_cohort[i])].produced;
        const int64_t live_chunk = trial ? trial->chunk() : measured_chunk > 0 ? measured_chunk
            : adaptive ? policy.tokens() : std::min(C, fixed_chunk);
        constexpr int64_t terminal_tokens = 256;
        const int64_t left = b0 - q;
        // Reserve a short tail for checkpoint/refill. The preceding partial slice
        // still gives the original decoders their normal budget and latency guard.
        const bool terminal_split = learning_scope && trial && trial->state() == Trial::State::promoted &&
                                    runtime_.batch_on() && left <= live_chunk && left > terminal_tokens;
        const int64_t r = terminal_split ? b0 - terminal_tokens
            : std::min(b0, q + (runtime_.batch_on() ? live_chunk : C));
        if (terminal_split)
            basalt::log::info("basalt batch", "overlap learning: reserve %lld terminal tokens after a %lld-token slice",
                              (long long) terminal_tokens, (long long) (r - q));
        const auto tq = Clock::now();
        if (measure_overlap && runtime_.batch_on()) {
            ++measured_slices;
            measured_tokens += r - q;
        }
        // each chunk runs as a prefill of its own here, so the next chunk's n-gram rows would wait for it: once
        // this chunk has its rows, the rows that follow it are read into the row cache beside it
        runtime_.sp.set_ple_warm(ids.data(), q, r, n - 1);
        if (!runtime_.sp.run(ids.data() + q, r - q, q, e)) return false;
        const auto prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
        const auto slice_tokens = r - q;
        runtime_.scheduler_chunk = slice_tokens;
        q = r;
        runtime_.observe_scheduler(runtime_.batch_on() ? QfnRuntime::SchedulerPhase::overlap
                                                      : QfnRuntime::SchedulerPhase::prefill, n - q);
        if (q >= b0) {
            if (adaptive && runtime_.batch_on()) {
                const auto now = Clock::now();
                double gap_ms = 0;
                for (const auto& slot : runtime_.bs) if (slot.active && slot.last_emit > slot.t0)
                    gap_ms = std::max(gap_ms, std::chrono::duration<double, std::milli>(now - slot.last_emit).count());
                basalt::log::info("basalt batch", "overlap terminal slice: %lld tokens, %.1f ms since latest decoder output",
                                  (long long) slice_tokens, gap_ms);
            }
            continue;
        }
        int ys = -1;
        {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
            std::lock_guard<std::mutex> lk(runtime_.in_mu);
            for (auto it = runtime_.in_lines.begin(); it != runtime_.in_lines.end();) {
                if (it->rfind("BSTOP ", 0) == 0) {
                    const int b = std::atoi(it->c_str() + 6);
                    if (b >= 0 && b < (int) runtime_.bs.size()) runtime_.bs[(size_t) b].stop = true;
                    it = runtime_.in_lines.erase(it);
                } else if (it->rfind("BYIELD ", 0) == 0) {
                    ys = std::atoi(it->c_str() + 7);
                    it = runtime_.in_lines.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (ys >= 0) {
            std::string ye;
            const auto ty = Clock::now();
            if (can_yield(q, b0, ys) && park_read(q, ys, ye)) {
                basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                     "holds it (copied in %.1f ms)", (long long) q, (long long) n, ys,
                             std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                e = "yield";
                return false;
            }
            basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld%s%s", ys, (long long) q,
                         (long long) n, ye.empty() ? "" : ": ", ye.c_str());
        }
        if (decode_share <= 0.0 || !runtime_.batch_on()) continue;
        const double budget = trial ? 0.50 * prefill_ms : measured_share > 0.0 ? measured_share * prefill_ms
            : adaptive_quantum ? policy.decode_budget(prefill_ms)
            : decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
        runtime_.scheduler_quantum_ms = budget;
        const auto td = Clock::now();
        double gap_ms = 0.0;
        int64_t decoded = 0;
        do {
            if (measure_overlap) ++measured_windows;
            if (!runtime_.batch_step(true)) { batch_fatal = true; e = "a batch window failed"; return false; }
            if (!adaptive) continue;
            const auto now = Clock::now();
            for (size_t b = 0; b < progress.size(); ++b) {
                const auto& slot = runtime_.bs[b];
                auto& last = progress[b];
                if (slot.produced == last.produced) continue;
                decoded += std::max<int64_t>(0, slot.produced - last.produced);
                const auto sent = runtime_.scheduler_enabled ? slot.last_emit : now;
                gap_ms = std::max(gap_ms, std::chrono::duration<double, std::milli>(sent - last.when).count());
                last = {slot.produced, sent};
            }
        } while (runtime_.batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
        if (runtime_.shared_pages && runtime_.mtp_on && !runtime_.mtp.use_main(e)) return false;
        if (adaptive && runtime_.batch_on()) {
            const auto now = Clock::now();
            for (size_t b = 0; b < progress.size(); ++b)
                if (runtime_.bs[b].active)
                    gap_ms = std::max(gap_ms, std::chrono::duration<double, std::milli>(now - progress[b].when).count());
            const double cycle_ms = std::chrono::duration<double, std::milli>(now - tq).count();
            if (trial) {
                const auto previous = trial->state();
                if (const char* reason = changed()) trial->interrupt(reason);
                else if (terminal_split) trial->check_terminal_gap(gap_ms);
                else {
                    for (size_t i = 0; i < trial_cohort.size(); ++i)
                        emitted[i] += runtime_.bs[std::get<0>(trial_cohort[i])].produced;
                    trial->observe(slice_tokens, cycle_ms, gap_ms, emitted);
                }
                const auto current = trial->state();
                if (current != previous) {
                    constexpr const char* decisions[] = {"overlap_calibrating", "overlap_probe", "overlap_confirming",
                                                         "overlap_promoted", "overlap_rolled_back", "overlap_interrupted"};
                    publish_trial(decisions[(int) current], trial->interruption_reason());
                    if (current == Trial::State::rejected) {
                        if (learning_scope) runtime_.reject_overlap_learning(learning_scope, candidate);
                        if (hinted) {
                            std::printf("SCHEDHINTDROP %016llx\n", (unsigned long long) runtime_.overlap_hint_scope);
                            std::fflush(stdout);
                            runtime_.overlap_hint_scope = 0;
                        }
                        basalt::log::info("basalt batch", "overlap trial rollback: cycle %.1f ms, gap %.1f ms, "
                                          "opening max gap %.1f ms, confirming max gap %.1f ms, active limit %.1f ms",
                                          cycle_ms, gap_ms, trial->opening_gap_ms(), trial->confirmed_gap_ms(), trial->gap_limit_ms());
                    }
                    if (current == Trial::State::rejected || current == Trial::State::interrupted) trial.reset();
                }
            } else {
                policy.observe(gap_ms, prefill_ms, runtime_.scheduler_enabled, decoded, cycle_ms);
                if (measured_chunk == 0 && policy.tokens() != live_chunk)
                    basalt::log::info("basalt batch", "interleaved prefill: chunk %lld -> %lld (decode gap %.1f ms, reference %.1f ms)",
                                      (long long) live_chunk, (long long) policy.tokens(), gap_ms, policy.reference_gap_ms());
            }
        }
        if (trial && !runtime_.batch_on()) {
            publish_trial("overlap_interrupted", "decoder_drained");
            trial.reset();
        }
        ++il_parts;
        const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
        il_ms += decode_ms;
        if (measure_overlap) {
            measured_budget_ms += budget;
            measured_decode_ms += decode_ms;
        }
        // The last decode window has settled. The caller can resize the loan and plan the unread suffix.
        if (solo_at != nullptr && !runtime_.batch_on() && b0 - q > std::min(C, live_chunk)) {
            // The last window can leave an expert exchange whose incoming write has not been issued.
            // Settle it before the solo layout borrows any of its destination slots.
            if (runtime_.ba_job) { runtime_.ba_thr.join(); runtime_.ba_job = false; }
            runtime_.apply_pending(true);
            if (!runtime_.xq_drain()) {
                batch_fatal = true;
                e = "prefill solo handoff: unable to settle adaptive expert swaps";
                return false;
            }
            *solo_at = q;
            return true;
        }
    }
    if (learning_scope && trial && trial->settled_winner() && !changed())
        overlap_learning_sample = OverlapLearningSample{learning_scope, candidate, trial->gap_limit_ms(),
                                                       std::move(trial_cohort)};
    if (measure_overlap)
        basalt::log::info("basalt batch", "tuner overlap: %lld tokens in %lld slices, %lld windows in %.1f ms, budget %.1f ms",
                          (long long) measured_tokens, (long long) measured_slices, (long long) measured_windows,
                          measured_decode_ms, measured_budget_ms);
    return true;
}

}  // namespace basalt::models::qwen4exp::serve
