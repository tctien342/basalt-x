// src/models/qwen4exp/serve/qfn_decode.cpp - Flash-Next's side of the window loop: verify windows, drafts, adaptive rounds.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "basalt/models/qwen4exp/decode/dflash2.hpp"
#include "../../../program/serve/decode_loop.hpp"

namespace basalt::models::qwen4exp::serve {

namespace {
// Tune proposal length without changing target verification or sampling.
float proposal_threshold(const basalt::core::DraftCumAdapt& adaptive) {
    static const float scale = basalt::core::exp_flag("mtp_longer_proposals") != 0 ? 0.85f : 1.0f;
    return adaptive.th() * scale;
}

// --draft-log FILE: one line per window (T, accepted, draft probabilities, ms) for draft-policy study
FILE* draft_log_file() {
    static FILE* const f = basalt::core::diagnostics().draft_log.empty() ? nullptr
        : std::fopen(basalt::core::diagnostics().draft_log.c_str(), "a");
    return f;
}
// spec_log: per window, the target's sampling distribution at each verified row beside the draft head's top-4
// of the draft that row checks, for offline tree-draft studies (written next to --draft-log as FILE.tk)
FILE* tk_log_file() {
    static FILE* const f = basalt::core::exp_flag("spec_log", 0) != 0 && !basalt::core::diagnostics().draft_log.empty()
        ? std::fopen((basalt::core::diagnostics().draft_log + ".tk").c_str(), "a") : nullptr;
    return f;
}
}  // namespace

QfnRequest::DecSnap QfnRequest::dec_snap() {
    return DecSnap{runtime_.ver.ms_wait, runtime_.ver.ms_pool, runtime_.ver.ms_host, runtime_.drive.d.ms_plan, runtime_.drive.d.ms_actq, runtime_.drive.d.ms_jobs,
                   runtime_.drive.d.ms_run, runtime_.drive.d.peer != nullptr ? runtime_.drive.d.peer->ms_wait : 0.0,
                   runtime_.drive.d.multi_misses, runtime_.drive.d.multi_entries, runtime_.drive.d.cache_hits, runtime_.drive.d.pcie_experts,
                   runtime_.drive.d.peer_entries};
}

Flow QfnRequest::decode_begin() {
    if (runtime_.mtp_on && !runtime_.mtp.use_main(runtime_.err)) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }   // the windows between prompt chunks swapped it
    if (!runtime_.peer_due_apply(runtime_.err)) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }
    runtime_.observe_scheduler(QfnRuntime::SchedulerPhase::solo_decode);
    return Flow::proceed;
}

void QfnRequest::seed_tiers() {
    {   // a request decoded here starts its adaptation from the experts its own prompt routed to
        std::vector<uint32_t> hist;
        int64_t hist_tokens = 0;
        const bool seeded = !runtime_.drive.d.usage.empty() && runtime_.sp.route_hist_take(hist, hist_tokens) && (!in_slot || solo_req);
        // a split read counted its stages' layers there
        if (seeded && runtime_.split_hist.size() == hist.size())
            for (size_t i = 0; i < hist.size(); ++i) hist[i] += runtime_.split_hist[i];
        runtime_.split_hist.clear();
        if (seeded && basalt::core::exp_flag("prefill_route_log") != 0 &&
            !basalt::core::diagnostics().draft_log.empty()) {
            static FILE* const f = std::fopen((basalt::core::diagnostics().draft_log + ".prefill").c_str(), "ab");
            if (f != nullptr) {
                const uint64_t header[] = {(uint64_t) hist_tokens, (uint64_t) hist.size()};
                std::fwrite(header, sizeof(header), 1, f);
                std::fwrite(hist.data(), sizeof(hist[0]), hist.size(), f);
                std::fflush(f);
            }
        }
        if (seeded) seed_usage_from_prompt(runtime_.drive.d.usage, hist, hist_tokens, runtime_.host_res, runtime_.g.n_expert, runtime_.peer);
    }
}

void QfnRequest::decode_state_begin() {
    drafts = std::vector<int32_t>((size_t) std::max(runtime_.S, basalt::core::kMtpChainMax), 0);
    dprob = std::vector<float>((size_t) std::max(runtime_.S, basalt::core::kMtpChainMax), 0.0f);
    sbuf = std::vector<int32_t>((size_t) runtime_.S, 0);
    if (runtime_.o.suffix_draft > 0) {
        const int64_t context = runtime_.o.max_context;
        const int64_t output = std::min<int64_t>(max_new, context);
        const size_t capacity = (size_t) std::min(context + 4096, n + output + 1);
        runtime_.sfx.emplace(std::max(1, runtime_.o.suffix_draft), 64, capacity);
        for (int64_t t : ids) runtime_.sfx->append((int32_t) t);
    }
    sfx_windows = 0;
    sfx_drafts = 0;
    sfx_ok = 0;
    // pipelined windows: `cv` runs the next window, `ov` is the other chain; a guess that held leaves the next
    // window's verdict in `held`
    cv = &runtime_.ver;
    max_thinking = req_think_close >= 0;
    thinking_mtp = max_thinking && runtime_.thinking_ver->max_t() > 2 &&
                   !geni && !req_bias_present && gguard == nullptr && ppl == nullptr;
    thinking_offered = thinking_accepted = thinking_close_clips = 0;
    ov = &runtime_.ver_b;

    held.win.assign((size_t) runtime_.S, 0);
    held.out.assign((size_t) runtime_.S, 0);
    win_b = std::vector<int32_t>((size_t) runtime_.S, 0);
    n_chain = 0;                       // drafts the last chain produced
    full_ema = 0.5;                 // how often windows are accepted whole
    pair_hint = false;                // windows are accepted whole often enough to run the chain on for a guess
    pair_req = runtime_.pipe_on && ppl == nullptr && !relax_on && gguard == nullptr && hist_n == 0 && !in_slot && req_logprobs < 0 &&
                          runtime_.kv_floor == 0;
    if (runtime_.pipe_on) runtime_.mtp.set_max_drafts((pair_req ? basalt::core::kMtpChainMax : runtime_.S_mtp) - 1);
    pair_runs = 0;
    pair_holds = 0;
}

void QfnRequest::decode_stats_begin() {
    // --decode-timing: where a request's decode time goes (one line per request)
    dec_timing = basalt::core::diagnostics().decode_timing;

    ds0 = dec_snap();
    dt_run = 0;
    dt_commit = 0;
    dt_draft = 0;
    dec_windows = 0;
    dec_T = 0;
    decode_hits0 = runtime_.drive.d.cache_hits;
    swapped0 = runtime_.xq_swapped;
    noplace0 = runtime_.xq_noplace;
    // the RAM and file tiers of this request (the mmap source; 0 with the arena)
    ram0 = runtime_.src.ram_reads();
    files0 = runtime_.src.file_reads();
    file_bytes0 = runtime_.src.file_read_bytes();
    decode_look0 = runtime_.drive.d.cache_hits + runtime_.drive.d.cache_admitted + runtime_.drive.d.cache_refused;
    offload0 = runtime_.drive.d.offload_entries;
    adapt_age = 2;   // loop tops since the last adaptive round
    applied_early = false;

    copy_live = false;
    copy_done = false;   // exchanges applied early whose RAM copies are under way
    draft_log = draft_log_file();
    if (draft_log != nullptr) {
        std::fprintf(draft_log, "R temp=%g S=%d\n", (double) req_temperature, runtime_.S_mtp);
        std::fflush(draft_log);
    }
    tk_log = tk_log_file();
    tk_valid = false;
    if (tk_log != nullptr) { std::fprintf(tk_log, "R temp=%g S=%d\n", (double) req_temperature, runtime_.S_mtp); std::fflush(tk_log); }
}

inline int QfnRequest::thinking_window_cap() const {
    return (int) std::min<int64_t>({runtime_.thinking_ver->max_t(), runtime_.S_mtp,
                                    runtime_.o.max_context - p, max_new - produced_n});
}

inline int QfnRequest::draft_window() {
    if (max_thinking) {
        if (!thinking_mtp) return 1;
        const int cap = thinking_window_cap();
        int T = 1;
        while (T < cap && basalt::core::draft_keep(dprob.data(), T - 1, proposal_threshold(cum_adapt))) {
            // The target can emit the close from the previous row. The K=10 verifier consumes it next.
            if (drafts[(size_t) T - 1] == req_think_close) {
                ++thinking_close_clips;
                break;
            }
            ++T;
        }
        return T;
    }
    if (dflash) return 1 + dflash_count;
    int T = 1;
        if (held.on) T = held.T;
        else
        while (T < runtime_.S_mtp && basalt::core::draft_keep(dprob.data(), T - 1, proposal_threshold(cum_adapt))) ++T;
    return T;
}

inline void QfnRequest::lookup_window(int& T, bool& from_sfx, int& sfx_match) {
        if (max_thinking && !thinking_mtp) return;
        if (dflash_suffix_window > 0) {
            T = dflash_suffix_window;
            from_sfx = true;
            sfx_match = dflash_suffix_match;
            dflash_suffix_window = 0;
            return;
        }
        if (runtime_.o.suffix_draft > 0 && !first_window && ppl == nullptr && !held.on) {
            const int k = runtime_.sfx->propose(runtime_.S - 1, sbuf.data());
            sfx_match = runtime_.sfx->last_match();
            if (k > 0 && ((dflash && T == 1) || sbuf[0] == drafts[0])) {
                const basalt::spec::DraftPolicy::Pick pk = runtime_.policy->choose(T, k, sfx_match);
                if (pk.lookup) {
                    int width = pk.t;
                    if (max_thinking) {
                        width = std::min(width, thinking_window_cap());
                        for (int i = 0; i < width - 1; ++i) {
                            if (sbuf[(size_t) i] != req_think_close) continue;
                            width = i + 1;
                            ++thinking_close_clips;
                            break;
                        }
                    }
                    if (width > 1) { T = width; from_sfx = true; }
                }
            }
        }
}

inline void QfnRequest::fill_drafts(int T, bool from_sfx) {
        for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
        if (held.on)
            for (int i = 0; i < T; ++i) window[(size_t) i] = held.win[(size_t) i];
}

inline int QfnRequest::before_window(int T, int64_t) {
    if (runtime_.scheduler_enabled) {
        runtime_.scheduler_rows = T;
        runtime_.observe_scheduler(QfnRuntime::SchedulerPhase::solo_decode);
    }
        runtime_.drive.d.layers = 0;
        runtime_.drive.d.experts = 0;
        runtime_.drive.d.failed = false;
        // the previous adapt round's copies land first: with a non-blocking query, whether a swapped-in expert
        // ran on the GPU or the CPU (they round differently) would depend on the copy's timing
        basalt::core::decode_gaps().mark(17);
        // an adaptive round's swaps land before the second window after it (a fixed window, so reproducible),
        // applied beside the draft before it (apply_thr) when their copies are done; what that moved to the
        // peer lands one window later
        if (apply_thr.joinable()) apply_thr.join();
        if (copy_live && copy_done) {   // the RAM copies are done: their bookkeeping, between windows
            basalt::core::decode_events().tag(1u << 3);
            runtime_.src.commit_exchanges();
            copy_live = false;
        }
        ++adapt_age;
        if (adapt_age >= 2 && !(applied_early && adapt_age == 2)) {
            if (!runtime_.pending.empty()) basalt::core::decode_events().tag(1u << 1);
            runtime_.apply_pending(true);
        }
        if (!held.on && (runtime_.vision_queued.load() || runtime_.vision_decode_lease.load())) {
            if (adapt_thr.joinable()) adapt_thr.join();
            if (copy_live) { runtime_.src.commit_exchanges(); copy_live = false; }
            if (!cv->wait_commit(runtime_.err) || !runtime_.service_vision()) return 2;
        }
        // the elastic KV: the next window's cells backed before it runs (between windows, rare)
        if (runtime_.kv_floor > 0 && p + T + 64 > basalt::core::session_kv_cells(*runtime_.rsp)) {
            if (adapt_thr.joinable()) adapt_thr.join();
            if (copy_live) { runtime_.src.commit_exchanges(); copy_live = false; }   // as at the request's end
            if (!runtime_.kv_fit(*runtime_.rsp, p + runtime_.kKvGrowCells, runtime_.err, true, p + runtime_.kKvWindowCells)) {
                if (runtime_.kv_short) {
                    kv_stop = runtime_.err;
                    return 1;
                }
                return 2;
            }
        }
        if (req_think_close >= 0) {
            auto* next = !first_window && max_thinking ? runtime_.thinking_ver.get() : &runtime_.ver;
            if (next != cv) {
                if (!cv->wait_commit(runtime_.err)) return 2;
                cv = next;
                if (max_thinking)
                    basalt::log::info("basalt serve", "thinking max: decode K=%d on the last %d layers at position %lld",
                                      runtime_.o.target_k_thinking_num, runtime_.o.target_k_last_layer_num, (long long) p);
                else
                    basalt::log::info("basalt serve", "thinking max: decode K=10 at position %lld", (long long) p);
            }
        }
        basalt::core::decode_gaps().mark(18);
    return 0;
}

inline void QfnRequest::spawn_adapt(bool early) {
        
            if (runtime_.adapt_fast) {   // a round, or the next step of the one in flight, beside the draft
                if (!do_adapt && runtime_.xq_stage == 0) return;
                if (do_adapt) adapt_age = 0;
                basalt::core::decode_events().tag(1u << 0);
                basalt::core::decode_events().tag(1u << (6 + runtime_.xq_stage));
                adapt_early = early;
                adapt_thr.run([&, early] {
                    auto ms = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
                    const auto ta = Clock::now();
                    basalt::core::decode_events().stamp(8);
                    std::vector<QfnRuntime::Swap> ranked;
                    const bool pre = early && runtime_.xq_stage == 0;
                    if (pre) runtime_.rank_swaps(ranked, runtime_.xq_across);
                    const double ms_pre = ms(ta);
                    basalt::core::SpinWait spin;
                    while (early && adapt_go.load(std::memory_order_acquire) == 0) spin();
                    if (early && adapt_go.load(std::memory_order_acquire) != 1) return;
                    const auto tb = Clock::now();
                    adapt_ok = runtime_.xq_stage != 0 ? runtime_.xq_advance(false) : runtime_.xq_round(pre ? &ranked : nullptr);
                    basalt::core::decode_events().stamp(6);
                    std::string perr;
                    if (adapt_ok && !cv->dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                    basalt::core::decode_events().stamp(7);
                    basalt::core::decode_gaps().hms[20] += ms_pre + ms(tb);
                });
                return;
            }
            if (!do_adapt) return;
            adapt_age = 0;
            adapt_thr.run([&] {
                const auto ta = Clock::now();
                adapt_ok = runtime_.adapt();
                if (adapt_ok) {   // the evictions' residency, uploaded beside the draft
                    std::string perr;
                    if (!cv->dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                }
                basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            });
        }

inline void QfnRequest::tier_before(int T, int e0, bool from_sfx) {
    // the guess: this window is accepted whole and the target's next token is the chain's draft T - 1; the next
    // window then holds that draft and the ones the chain made after it
        try_pair = pair_req && pair_hint && !held.on && !first_window && !from_sfx && e0 == 0 &&
                              p + T + runtime_.S_mtp + 64 < runtime_.o.max_context;
        // the window's adaptive step, decided before it runs (the window changes none of its inputs)
        adapt_ok = true;
        do_adapt = !runtime_.drive.d.usage.empty() && ((runtime_.rounds + 1) % runtime_.o.adapt_every) == 0 &&
                              (runtime_.adapt_fast ? runtime_.xq_stage == 0
                                          : (runtime_.pending.empty() && !copy_live && !runtime_.src.exchanges_staged()));
    adapt_early = false;
    adapt_go.store(0, std::memory_order_relaxed);   // 1 go, 2 dropped (the window failed)
    (void) e0;
    (void) from_sfx;
        // the plain rounds copy into the slots at once, so they start after the window; a paired window is held or not
        // only once it has run
        if (runtime_.adapt_fast && !held.on && !try_pair) cv->set_served_hook([this] { spawn_adapt(true); });
}

inline void QfnRequest::tier_after_commit() {
            if (adapt_early) adapt_go.store(1, std::memory_order_release);
            else spawn_adapt(false);
            // the swaps' residency at the next window, then their RAM copies a share per draft
            applied_early = adapt_age == 1 && !do_adapt && !runtime_.pending.empty();
            if (applied_early || copy_live) {
                const bool boundary = applied_early, copies = applied_early || copy_live;
                if (copies) {
                    copy_live = true;
                    copy_done = false;
                }
                basalt::core::decode_events().tag(1u << 2);
                apply_thr.run([&, boundary, copies] {
                    if (boundary) {
                        runtime_.apply_pending(true, false);
                        std::string perr;
                        if (!cv->dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                    }
                    if (copies) copy_done = !runtime_.src.exchanges_staged() || runtime_.src.copy_exchanges(8ull << 20);
                });
            }
        }

inline void QfnRequest::tier_finish() {
    int wait = 0;
    if (adapt_early) adapt_go.compare_exchange_strong(wait, 2);
    if (adapt_thr.joinable()) adapt_thr.join();
}

inline bool QfnRequest::run_window(int T, int e0, int& TB) {
        if (cv == runtime_.thinking_ver.get()) {
            auto* plan = runtime_.drive.d.plan;
            runtime_.drive.d.plan = cv->plan_sink();
            const bool ok = cv->run(T, window.data(), p, runtime_.win_pool_fn, runtime_.win_pool_user,
                                    outv.data(), runtime_.err);
            runtime_.drive.d.plan = plan;
            return ok;
        }
        const std::function<int(int32_t*)> guess = [&](int32_t* wb) -> int {
            n_chain = runtime_.mtp.draft_more(n_chain, drafts.data(), dprob.data());   // made beside this window's stage 0
            if (n_chain <= T || !basalt::core::draft_keep(dprob.data(), T - 1, kPairTheta)) return 0;
            int tb = 1;
            while (tb < runtime_.S_mtp && T - 1 + tb < n_chain && basalt::core::draft_keep(dprob.data() + T, tb - 1, proposal_threshold(cum_adapt)))
                ++tb;
            if (tb < 2) return 0;
            wb[0] = drafts[(size_t) T - 1];
            for (int i = 1; i < tb; ++i) wb[i] = drafts[(size_t) T - 1 + (size_t) i];
            return tb;
        };
        user_c = cv == &runtime_.ver ? runtime_.win_pool_user : (void*) &runtime_.split_drive_b;
        user_o = cv == &runtime_.ver ? (void*) &runtime_.split_drive_b : runtime_.win_pool_user;
        // the prompt rows the first window folds in send no pick: the head scores them only roughly (exact logits
        // are kept where something reads them: a scored chunk, the logit-hash diagnostic, and a sampled window's
        // kept lists, which the drafter notes as candidates)
        if (first_window && e0 > 0 && ppl == nullptr && !runtime_.logit_hash && req_temperature <= 0.0f)
            cv->set_head_from(e0);
        const bool ran_ok = held.on ? true
            : try_pair ? cv->run_pair(*ov, T, window.data(), p, guess, win_b.data(), TB, runtime_.win_pool_fn, user_c, user_o,
                                      outv.data(), runtime_.err)
                       : cv->run(T, window.data(), p, runtime_.win_pool_fn, user_c, outv.data(), runtime_.err);
    return ran_ok;
}

inline bool QfnRequest::take_held(int T) {
        const bool held_run = held.on;
        if (held.on) {
            for (int i = 0; i < T; ++i) outv[(size_t) i] = held.out[(size_t) i];
            held.on = false;
        }
    return held_run;
}

inline void QfnRequest::relax_note_forced(int a) {
            if (relax_on) {
                if (outv[(size_t) a] == runtime_.o.relax_call_open) in_call = true;
                else if (outv[(size_t) a] == runtime_.o.relax_call_close) in_call = false;
            }
}

inline int QfnRequest::accept_drafts(int T) {
    int a = 0;
    // Keep the target sampler authoritative inside widened thinking.
    if (relax_on && !max_thinking) {
            // a draft the target's draw disagrees with is kept when the target gives it probability >= P,
            // outside tool calls and never for an added token; the row then emits the draft
            bool call = in_call;
            while (a < T - 1) {
                const int32_t d = window[(size_t) a + 1];
                if (d != outv[(size_t) a]) {
                    if (call || d >= runtime_.o.relax_special_from || cv->draft_prob(a) < req_relax) break;
                    outv[(size_t) a] = d;
                    ++relaxed_kept;
                }
                if (d == runtime_.o.relax_call_open) call = true;
                else if (d == runtime_.o.relax_call_close) call = false;
                ++a;
            }
    } else {
        while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
    }
    if (relax_on) {
            for (int i = 0; i <= a; ++i)
                if (outv[(size_t) i] == runtime_.o.relax_call_open) in_call = true;
                else if (outv[(size_t) i] == runtime_.o.relax_call_close) in_call = false;
    }
    return a;
}

inline bool QfnRequest::after_accept(int T, int& a, int e0, bool from_sfx, bool held_run, bool paired, bool& hold) {
        // the guess held when the window was accepted whole, its last pick is the next window's first token, and
        // the request goes on past it
        if (!from_sfx && e0 == 0 && !held_run) full_ema = 0.9 * full_ema + (a == T - 1 ? 0.1 : 0.0);
        if (paired) {
            ++pair_runs;
            int n_emit_h = 0;
            bool eos_h = false;
            for (int i = e0; i <= a && produced_n + n_emit_h < max_new && !eos_h; ++i) {
                ++n_emit_h;
                eos_h = std::find(runtime_.o.eos_ids.begin(), runtime_.o.eos_ids.end(), (int64_t) outv[(size_t) i]) != runtime_.o.eos_ids.end();
            }
            hold = a == T - 1 && outv[(size_t) a] == win_b[0] && !eos_h && produced_n + n_emit_h < max_new && !runtime_.halted();
            if (!hold && !cv->pair_miss(*ov, runtime_.win_pool_fn, user_o, runtime_.err)) {
                return false;
            }
        }
        if (runtime_.logit_hash)   // the accepted rows' logits, per position (equality of the batch rows with these)
            for (int t = 0; t <= a; ++t) basalt::log::raw("LH -1 %lld %016llx\n", (long long) (p + t), (unsigned long long) runtime_.row_hash(t));
        if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
        if (tk_log != nullptr && tk_valid && !from_sfx && ppl == nullptr && req_temperature > 0.0) {
            std::vector<float> lg((size_t) runtime_.n_vocab);
            std::fprintf(tk_log, "W %d %d\n", T, a);
            for (int j = 0; j < T; ++j) {
                if (j > a) {   // past the first miss: only the draft side
                    std::fprintf(tk_log, "R %d - - T D", j);
                    for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                    std::fputc('\n', tk_log);
                    continue;
                }
                if (!cv->copy_logits(j, lg.data())) break;
                // top_k, top_p, then temperature over the survivors: the sampler chain without penalties
                std::vector<int> ord((size_t) runtime_.n_vocab);
                for (int i = 0; i < runtime_.n_vocab; ++i) ord[(size_t) i] = i;
                const int K = basalt::kernels::sampler_list_len(req_top_k, (int) runtime_.n_vocab);
                std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int x, int y) {
                    return lg[(size_t) x] > lg[(size_t) y] || (lg[(size_t) x] == lg[(size_t) y] && x < y);
                });
                const double mx = lg[(size_t) ord[0]];
                std::vector<double> e((size_t) K);
                double s1 = 0.0;
                for (int i = 0; i < K; ++i) s1 += (e[(size_t) i] = std::exp((double) lg[(size_t) ord[(size_t) i]] - mx));
                int keep = K;
                double cum = 0.0;
                for (int i = 0; i < K; ++i) { cum += e[(size_t) i] / s1; if (cum >= (double) req_top_p) { keep = i + 1; break; } }
                double s2 = 0.0;
                for (int i = 0; i < keep; ++i) s2 += (e[(size_t) i] = std::exp(((double) lg[(size_t) ord[(size_t) i]] - mx) / (double) req_temperature));
                std::fprintf(tk_log, "R %d %d %d", j, j + 1 < T ? (int) window[(size_t) j + 1] : -1, (int) outv[(size_t) j]);
                std::fprintf(tk_log, " T");
                for (int i = 0; i < std::min(keep, 8); ++i) std::fprintf(tk_log, " %d:%.5f", ord[(size_t) i], e[(size_t) i] / s2);
                std::fprintf(tk_log, " D");
                for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                std::fputc('\n', tk_log);
            }
            std::fflush(tk_log);
        }
        tk_valid = false;
    return true;
}

inline void QfnRequest::note_draft_probs(bool from_sfx) {
    wprob_n = std::min(runtime_.S_mtp, 16);
    if (draft_log != nullptr)
        for (int i = 0; i < wprob_n; ++i) wprob[i] = first_window || from_sfx ? -1.0f : dprob[(size_t) i];
}

template <class After>
inline bool QfnRequest::draft_next(int T, int a, int e0, int TB, bool from_sfx, bool hold, bool will_draft, After& after_launch) {
    basalt::core::decode_gaps().mark(5);
    if (auto* draft = dflash) {
        dflash_count = 0;
        if (ppl != nullptr) return true;
        static const bool suffix_first = basalt::core::exp_flag("dflash_suffix_first") != 0;
        static const bool overlap_commit = basalt::core::exp_flag("dflash_overlap_commit") != 0;
        const bool defer_commit = overlap_commit && !suffix_first && will_draft && produced_n > 0;
        if (!defer_commit) after_launch();
        if (!draft->seed_prompt(p, runtime_.err) ||
            !draft->inject_verified(p, dflash_commit, runtime_.err)) return false;
        if (!will_draft) return draft->finish(runtime_.err);
        if (suffix_first && runtime_.o.suffix_draft > 0) {
            // after_launch has committed the current outputs into the suffix history.
            const int k = runtime_.sfx->propose(runtime_.S - 1, sbuf.data());
            const int match = runtime_.sfx->last_match();
            if (k >= 3 && match >= 8) {
                const auto pick = runtime_.policy->choose(std::min(runtime_.S, 4), k, match);
                if (pick.lookup && pick.t >= 4) {
                    dflash_suffix_window = pick.t;
                    dflash_suffix_match = match;
                    ++dflash_suffix_skipped;
                    return draft->finish(runtime_.err);
                }
            }
        }
        static const bool cost_based = basalt::core::exp_flag("dflash_cost_policy") != 0;
        const auto* costs = cost_based && runtime_.policy ? &*runtime_.policy : nullptr;
        const auto emit = defer_commit ? +[](void* p) { (*static_cast<After*>(p))(); } : nullptr;
        return draft->draft(outv[(size_t) a], p + dflash_commit, std::min(runtime_.S - 1, basalt::core::Dflash2::kDrafts),
                            req_sp, costs, drafts.data(), dflash_count, runtime_.err, emit, &after_launch);
    }
    const bool thinking_window = max_thinking;
    if (thinking_window && thinking_mtp) {
        thinking_offered += T - 1 - e0;
        thinking_accepted += a - e0;
    }
    if (thinking_window && (!thinking_mtp || outv[(size_t) a] == req_think_close)) {
        // Keep the committed prefix. Proposals from before the closing token cannot cross the K handoff.
        std::fill(dprob.begin(), dprob.end(), 0.0f);
        draft_failed = will_draft && runtime_.mtp_on &&
                       !runtime_.mtp.prefill(cv->final_R_all(), outv.data(), a + 1, p, runtime_.err);
        if (draft_failed) std::printf("ERR %s\n", runtime_.err.c_str());
        return !draft_failed;
    }
    if ((!thinking_window || thinking_mtp) && !from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
    runtime_.mtp.set_cum_th(proposal_threshold(cum_adapt));
    bool drafted = true;
    draft_failed = false;
        if (hold) {
            // the tokens go out; the next window's last stage runs behind this window's commit while its first
            // stage finishes, and the drafter (on the first card) takes this window's rows once that card is
            // free
            ++pair_holds;
            if (!cv->pair_hold_commit(T, runtime_.err)) {
                std::printf("ERR %s\n", runtime_.err.c_str());
                draft_failed = true;
                return false;
            }
            after_launch();
            const float* staged = nullptr;
            drafted = runtime_.mtp.stage_rows(cv->final_R_all(), T, &staged, runtime_.err);   // before the last card is busy again
            if (!drafted || !cv->pair_hold_begin(*ov, T, p, TB, win_b.data(), runtime_.err) ||
                !cv->pair_hold_finish_first(*ov, runtime_.win_pool_fn, user_o, runtime_.err) ||
                !(drafted = runtime_.mtp.prefill(staged, outv.data(), T, p, runtime_.err))) {
                std::printf("ERR %s\n", runtime_.err.c_str());
                draft_failed = true;
                return false;
            }
            if (!cv->pair_hold_finish(*ov, runtime_.win_pool_fn, user_o, held.out.data(), runtime_.err) || runtime_.drive.d.failed) {
                std::printf("ERR %s\n", runtime_.drive.d.failed && runtime_.drive.d.fail ? runtime_.drive.d.fail : runtime_.err.c_str());
                draft_failed = true;
                return false;
            }
            held.on = true;
            held.T = TB;
            for (int i = 0; i < TB; ++i) held.win[(size_t) i] = win_b[(size_t) i];
            std::swap(cv, ov);
        } else if (will_draft && runtime_.mtp_on) {
            runtime_.mtp.set_launch_hook(after_launch);
            pair_hint = pair_req && full_ema >= kPairFull;
            runtime_.mtp.set_cum_th(pair_hint ? proposal_threshold(cum_adapt) * kPairChainScale : proposal_threshold(cum_adapt));
            runtime_.mtp.set_host_th(proposal_threshold(cum_adapt));
            if (const int32_t* top = cv->top_ids())
                runtime_.mtp.note_candidates(top, (a + 1) * basalt::core::Verifier::kTopIds);
            drafted = runtime_.mtp.draft_from(cv->final_R_all(), T, outv.data(), p, a, drafts.data(), runtime_.err, dprob.data(), &n_chain);
            runtime_.mtp.set_launch_hook(nullptr);
            if (drafted && req_think_close >= 0 && !thinking_window && n_chain > 0 && !max_mtp_reported) {
                basalt::log::info("basalt serve", "thinking max: MTP resumed with %d proposals", n_chain);
                max_mtp_reported = true;
            }
            if (drafted && runtime_.pipe_on) {   // the next window's first stage starts after the chain's kernels
                const cudaEvent_t ev = runtime_.mtp.done_event();
                runtime_.ver.set_wait_event(ev);
                runtime_.ver_b.set_wait_event(ev);
            }
            if (tk_log != nullptr && drafted) {
                for (int j = 0; j < 16 && j < runtime_.S_mtp; ++j) {
                    tk_prev[j].n = runtime_.mtp.diag_top(j, 4, tk_prev[j].id, tk_prev[j].p);
                }
                tk_valid = true;
            }
        }
    return drafted;
}

inline void QfnRequest::window_timing(int T, int a, bool from_sfx, Clock::time_point round0, Clock::time_point tw0,
                                      Clock::time_point tw1, Clock::time_point tw2) {
    basalt::core::decode_events().stamp(0);
        {
            const Clock::time_point tw3 = Clock::now();
            auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
            dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
            ++dec_windows; dec_T += T;
            if (draft_log != nullptr) {
                std::fprintf(draft_log, "W %d %d %d %.3f %.3f %.3f", T, a, from_sfx ? 1 : 0, msd(round0, tw0),
                             msd(tw0, tw1), msd(tw1, tw3));
                for (int i = 0; i < wprob_n; ++i) std::fprintf(draft_log, " %.4f", wprob[i]);
                std::fputc('\n', draft_log);
            }
        }
}

void QfnRequest::decode_end() {
    if (thinking_mtp)
        basalt::log::info("basalt serve", "thinking max: inner MTP accepted %lld/%lld, close clips %lld",
                          (long long) thinking_accepted, (long long) thinking_offered, (long long) thinking_close_clips);
    if (apply_thr.joinable()) apply_thr.join();
    if (copy_live) runtime_.src.commit_exchanges();   // swaps applied early: the rest of their copies
    copy_live = false;
}

Flow QfnRequest::decode() { return decode_loop(*this); }

}  // namespace basalt::models::qwen4exp::serve
