#include "basalt/core/direct_peer.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include <atomic>
// src/models/qwen4exp/serve/prompt_split.cpp - request prompt_split operations.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"

namespace basalt::models::qwen4exp::serve {

QfnRequest::SplitPlan QfnRequest::split_plan(int64_t a0, int64_t b0, const char* no_split) {
    SplitPlan plan;
    if (!runtime_.split_ready) return plan;   // (the start line said why)
    // every part longer than a chunk that is not read across says why, with its numbers
    auto decline = [&](const std::string& why) {
        basalt::log::info("basalt serve", "prompt split: %lld tokens read on CUDA0 (%s)", (long long) (b0 - a0),
                          why.c_str());
    };
    const char* skip = no_split != nullptr ? no_split : runtime_.batch_on() ? "slots are decoding" : nullptr;
    if (skip != nullptr) {
        if (b0 - a0 > runtime_.read_cap()) decline(skip);
        return plan;
    }
    const engine_local::SplitPricing pr = runtime_.split_price(a0, b0, n - 1, runtime_.split_stages);
    if (!pr.why.empty()) decline(pr.why);
    if (pr.chunk <= 0) return plan;   // (no reason: the part is one chunk, not longer than one)
    if (pr.chunk > runtime_.split_chunk_max) {
        decline("chunks of " + std::to_string(pr.chunk) + " tokens, above the stages' " +
                std::to_string(runtime_.split_chunk_max));
        return plan;
    }
    if (pr.price.stages < 2) {
        decline("no stage set fits the cards' layer windows");
        return plan;
    }
    basalt::log::info("basalt serve", "prompt split: %lld tokens priced %.0f ms on CUDA0, %.0f ms with %s%s; read %s",
                      (long long) (b0 - a0), pr.price.tier_ms, pr.price.split_ms,
                      runtime_.split_layers_text(pr.price.stages, pr.price.cuts).c_str(),
                      runtime_.split_stages_text(pr.price).c_str(), pr.price.split ? "across" : "on CUDA0");
    if (!pr.price.split) return plan;
    auto cuts = pr.price.cuts;
    for (int k = 1; k < pr.price.stages; ++k) {
        PromptStage& st = *runtime_.pstages[(size_t) (pr.price.stages - 1 - k)];
        const int64_t l1 = k + 1 < pr.price.stages ? pr.price.cuts[(size_t) k] : runtime_.g.n_layers;
        const basalt::core::OnDevice on(st.dev);
        uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
        int& cut = cuts[(size_t) k - 1];
        auto fits = [&]() { return runtime_.split_region(st, b0 + 64, pr.chunk, cut, l1, w_at, s_at, b_at); };
        if (!fits()) {
            // Only a two-stage split can move its cut without changing another peer's range.
            const int interval = (int) std::max<int64_t>(1, runtime_.g.qsa_interval);
            const int original = cut;
            if (pr.price.stages == 2) {
                // Keep at least one full attention group on the peer.
                for (cut += interval; cut <= l1 - interval; cut += interval) {
                    if (fits()) {
                        basalt::log::info("basalt serve", "prompt split: peer capacity moves cut %d -> %d on CUDA%d",
                                          original, cut, st.dev);
                        break;
                    }
                }
            }
            if (cut != original && cut <= l1 - interval) continue;
            basalt::log::info("basalt serve", "prompt split: %lld tokens do not fit CUDA%d's slots; read on CUDA0",
                              (long long) (b0 - a0), st.dev);
            return plan;
        }
    }
    plan.chunk = pr.chunk;
    plan.head = pr.head;
    plan.stages = pr.price.stages;
    plan.cuts = std::move(cuts);
    return plan;
}

// [a0, b0) across the cards in chunks of `chunk` tokens: the main card reads layers [0, cuts[0]) of each chunk
// and hands it on, each stage reads its layers of it and hands it on while the cards before read the next; the
// main card keeps the last chunk and reads its later layers once the stages' state is back.  Main's loan is
// made by `lend`.
// `ran` stays false when it stopped before reading anything (the caller reads the part on the main card)
bool QfnRequest::read_split(int64_t a0, int64_t b0, const SplitPlan& plan, bool& ran, std::string& e) {
    ran = false;
    if (!runtime_.service_vision(true)) {
        basalt::log::error("basalt serve", "peer vision lease: %s", runtime_.err.c_str());
        std::exit(1);
    }
    const auto t0 = Clock::now();
    const int64_t n_tok = b0 - a0, chunk = plan.chunk, head = plan.head;
    const int64_t lb = plan.cuts[0];
    // the last chunk's first cell, from the chunks the main card reads (the one it keeps)
    const std::vector<int64_t> part_chunks = serve_loans::chunk_list(n_tok, chunk, head);
    const int64_t c_last = b0 - (part_chunks.empty() ? 0 : part_chunks.back());
    if (runtime_.sp.chunk() != chunk) {
        e = "prompt split: the main card's prompt buffers are not laid out for the split's chunk";
        return false;
    }
    // the stages in pipeline order, each with its layers
    std::vector<PromptStage*> run;
    for (int k = 1; k < plan.stages; ++k) {
        PromptStage* st = runtime_.pstages[(size_t) (plan.stages - 1 - k)].get();
        st->l0 = plan.cuts[(size_t) k - 1];
        st->l1 = k + 1 < plan.stages ? plan.cuts[(size_t) k] : runtime_.g.n_layers;
        run.push_back(st);
    }
    // Temporary prompt stages use a device-local copy of the request's image positions.
    // The main stage alone injects the image embeddings before it hands the hidden rows on.
    std::vector<std::shared_ptr<int32_t>> image_positions;
    if (!runtime_.req_imgs.empty()) {
        for (PromptStage* st : run) {
            const basalt::core::OnDevice on(st->dev);
            const size_t bytes = runtime_.mrope_host.size() * sizeof(int32_t);
            const int32_t* previous = basalt::kernels::mrope_table();
            int32_t* table = nullptr;
            if (cudaMalloc(&table, bytes) != cudaSuccess) {
                e = "prompt split: the peer image position table does not fit";
                return false;
            }
            image_positions.emplace_back(table, [dev = st->dev, previous](int32_t* p) {
                const basalt::core::OnDevice on(dev);
                (void) cudaDeviceSynchronize();
                basalt::kernels::mrope_table_set(previous);
                (void) cudaFree(p);
            });
            int can_peer = 0;
            // Device selection enables peer access once at startup for every supported pair.
            const bool direct = cudaDeviceCanAccessPeer(&can_peer, st->dev, 0) == cudaSuccess && can_peer &&
                                cudaMemcpyPeerAsync(table, st->dev, runtime_.d_mrope, 0, bytes, st->stream) == cudaSuccess;
            if (!direct) (void) cudaGetLastError();
            if ((direct ? cudaStreamSynchronize(st->stream) :
                          cudaMemcpy(table, runtime_.mrope_host.data(), bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
                e = "prompt split: the peer image position upload failed";
                return false;
            }
            basalt::kernels::mrope_table_set(table);
            basalt::log::info("basalt serve", "prompt split: image positions copied to CUDA%d via %s",
                              st->dev, direct ? "CUDA peer copy" : "host upload");
        }
    }
    PromptStage& last = *run.back();
    std::vector<basalt::prefill::PrefillStats> st0;
    for (PromptStage* st : run) st0.push_back(st->sp.stats());
    const basalt::prefill::PrefillStats sm0 = runtime_.sp.stats();
    // the tiers' swaps land first: no copy may still write a slot a stage takes
    for (PromptStage* st : run) {
        st->tier->apply_pending(true);
        if (st->tier_at == 0) runtime_.peer_landing = false;
        else runtime_.xlanding[st->tier_at - 1] = 0;
    }
    auto give_back_now = [&](PromptStage& st) {
        std::string ge;
        if (st.tier == &runtime_.peer && !runtime_.close_vision_tail(ge)) {
            basalt::log::error("basalt serve", "%s", ge.c_str());
            std::exit(1);
        }
        const basalt::core::OnDevice on(st.dev);
        (void) cudaDeviceSynchronize();
        (void) cudaGetLastError();
        (void) (st.tier->give_back(st.lent, ge) && st.tier->give_back_wait(st.lent, ge));
    };
    auto give_back_all = [&]() {
        for (PromptStage* st : run) give_back_now(*st);
    };
    void* vision_arena = nullptr;
    size_t vision_bytes = 0;
    for (PromptStage* stp : run) {
        PromptStage& st = *stp;
        const basalt::core::OnDevice on(st.dev);
        const basalt::core::ExpertCache& pc = st.tier->cache();
        uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
        if (st.l0 >= st.l1 || !runtime_.split_region(st, b0 + 64, chunk, st.l0, st.l1, w_at, s_at, b_at)) {
            e = "prompt split: too few tier slots for the prompt";
            give_back_all();
            return false;
        }
        if (const size_t bytes = runtime_.vision_stage_bytes(*st.tier)) {
            vision_arena = w_at - bytes;
            vision_bytes = bytes;
        }
        uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
        st.lent_slots = serve_loans::cache_slots_for(pc, (uint64_t) (base + pc.bytes() - b_at));
        const int32_t first = (int32_t) (pc.slots() - st.lent_slots);
        if (!st.tier->lend(first, st.lent, e)) {
            give_back_all();
            return false;
        }
        if (st.tier_at == 0) runtime_.peer_due_lent(st.lent);
        st.buf = base + (pc.slot_offsets() ? (size_t) pc.slot_offsets()[first]
                                           : (size_t) first * basalt::kernels::cpu::expert_layout().max_blob);
        st.buf_bytes = (uint64_t) (s_at - st.buf);
        // The peer cache is segmented even without elastic VRAM. Grant CUDA0
        // access to the complete mapped segments covered by this stage's loan.
        if (pc.segmented() && pc.segment_bytes() > 0 && basalt::core::direct_peer_enabled(0, st.dev)) {
            const uint64_t seg = (uint64_t) pc.segment_bytes();
            const uint64_t lo = (uint64_t) (st.buf - base) / seg * seg;
            const uint64_t end = (uint64_t) pc.mapped_bytes();
            // CUDA permissions belong to each segment mapping and disappear on unmap.
            for (uint64_t at = lo; at < end; at += seg)
                if (!basalt::core::direct_peer_grant(base + at, (size_t) std::min(seg, end - at), 0)) {
                    basalt::log::warn("basalt P2P", "CUDA%d stage loan access refused at offset %llu", st.dev,
                                      (unsigned long long) at);
                    break;
                }
        }
        // the stage's experts in the tier: its layers' that kept their slots
        const std::vector<int32_t>& r = st.tier->residency();
        std::fill(st.res.begin(), st.res.end(), basalt::core::kNotResident);
        std::copy(r.begin() + (std::ptrdiff_t) (st.l0 * runtime_.g.n_expert), r.begin() + (std::ptrdiff_t) (st.l1 * runtime_.g.n_expert),
                  st.res.begin() + (std::ptrdiff_t) (st.l0 * runtime_.g.n_expert));
        if (cudaMemcpyAsync(w_at, st.image, (size_t) st.image_bytes, cudaMemcpyHostToDevice, st.stream) !=
            cudaSuccess) {
            e = std::string("prompt split: the weights' upload: ") + cudaGetErrorString(cudaGetLastError());
            give_back_all();
            return false;
        }
        basalt::core::prompt_session_clear(st.ss, runtime_.g, (void*) st.stream);
    }
    // a part after the start of the context: each stage's layers' state so far, every cell (after the clear),
    // all cards at once
    if (a0 > 0) {
        bool ok = true;
        for (PromptStage* st : run) {
            const basalt::core::OnDevice on(st->dev);
            ok = ok && cudaStreamSynchronize(st->stream) == cudaSuccess &&
                 basalt::core::session_copy_state(st->ss, *runtime_.rsp, runtime_.g, st->l0, st->l1, 0, a0,
                                                  [&](void* d, const void* sr, uint64_t nb) {
                                                      return st->state_in.copy(d, sr, nb, e);
                                                  }, e);
        }
        for (PromptStage* st : run) ok = ok && st->state_in.wait(e);
        if (!ok) {
            if (e.empty()) e = "prompt split: a stage's state from the main session";
            give_back_all();
            return false;
        }
    }
    // (relayout waits for the stream: the weights and the session are in place after it); each stage runs alone
    // until the chain is set below
    for (PromptStage* st : run) {
        const basalt::core::OnDevice on(st->dev);
        st->sp.set_stage(st->l0, st == &last ? -1 : st->l1, nullptr);
        if (!st->sp.relayout(chunk, st->buf, st->buf_bytes, e)) {
            give_back_all();
            return false;
        }
    }
    const double ms_setup = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    ran = true;
    // the main card: layers [0, lb) of every chunk, handing each but the last on; the tier cards compute no
    // rows of the main card's layers meanwhile
    runtime_.sp.set_peer_share(false);
    runtime_.sp.set_stage(0, lb, &run[0]->sp);
    runtime_.sp.set_keep_last(true);
    for (PromptStage* st : run) {
        st->sp.should_stop = runtime_.sp.should_stop;
        if (!runtime_.drive.d.usage.empty()) st->sp.route_hist_reset();   // its layers' routed entries, counted apart
    }
    // `BYIELD <slot>` (a shorter request waits): at the end of a chunk's first layers the main card keeps that
    // chunk as the last one and reads its later layers once the stages' state is back; the slot then holds
    // the part read up to that chunk's end.  `c_cut`: the kept chunk's first cell.
    int64_t main_done = a0, yield_q = -1;
    int yield_ys = -1;
    std::atomic<int64_t> c_cut{c_last};
    std::atomic<int64_t> last_end{a0};   // the end of the last chunk the last stage read
    runtime_.sp.keep_here = [&](int64_t done) -> bool {
        if (done >= c_last) return false;
        int ys = -1;
        {
            std::lock_guard<std::mutex> lk(runtime_.in_mu);
            for (auto it = runtime_.in_lines.begin(); it != runtime_.in_lines.end();) {
                if (it->rfind("BYIELD ", 0) == 0) {
                    ys = std::atoi(it->c_str() + 7);
                    it = runtime_.in_lines.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (ys < 0) return false;
        if (can_yield(done, b0, ys)) {
            yield_q = done;
            yield_ys = ys;
            c_cut.store(main_done);
            return true;
        }
        basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld", ys, (long long) done,
                          (long long) n);
        return false;
    };
    for (size_t k = 0; k + 1 < run.size(); ++k) run[k]->sp.set_stage(run[k]->l0, run[k]->l1, &run[k + 1]->sp);
    // the last stage, after each chunk: the rows the draft layer reads, back into the hand-off buffer the chunk
    // came in (the card before writes it again two chunks later, and not at all for the last); after its last
    // chunk, every stage's state to the main session and the slots' experts back, while the main card reads
    // its own last chunk
    const int64_t first_cell = runtime_.mtp_on ? std::max<int64_t>(0, runtime_.mtp.first_needed()) : INT64_MAX;
    bool given = false;
    int64_t given_at = -1;   // the cell the stages' state came back up to
    double ms_copy = 0;
    std::future<bool> refill_issue;   // the tier slots' copies, queued on a thread of their own
    std::string refill_err;
    // the final rows of the last stage's chunks the draft layer reads, in the hand-off buffers they came in
    struct Relay { int64_t cell0, n; const float* rows; };
    std::vector<Relay> relays;
    const bool draft_fp16 = basalt::core::exp_flag("draft_relay_fp16") != 0;
    auto relay_copy = [&](float* dst, const float* src, int64_t count, std::string& ce) {
        if (dst == nullptr) return false;
        if (draft_fp16) {
            if (!last.sp.copy_draft_rows_fp16(src, dst, count, ce)) return false;
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true))
                basalt::log::info("basalt prefill", "draft relay FP16 (%zu bytes)", (size_t) count * sizeof(uint16_t));
            return true;
        }
        return cudaMemcpy(dst, src, (size_t) count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    // those of the last two chunks handed on stay where they came in; an earlier chunk's go past `chunk` rows
    // in the card before's two hand-off buffers, which no chunk of this read writes
    int64_t in_place_from = a0;
    {
        const std::vector<int64_t> cl = serve_loans::chunk_list(n_tok, chunk, head);
        int64_t at = a0;
        for (size_t i = 0; i + 1 < cl.size(); ++i) {
            if (i + 3 == cl.size()) in_place_from = at;
            at += cl[i];
        }
    }
    const basalt::prefill::Prefill& before_last = run.size() >= 2 ? run[run.size() - 2]->sp : runtime_.sp;
    const int64_t aside_rows = std::max<int64_t>(0, before_last.hand_rows() - chunk);
    int aside_buf = 0;
    int64_t aside_at = 0;
    std::mutex relay_mu;
    // mid-prompt checkpoints at the positions one card takes them: the main card saves its state when it
    // hands such a chunk on, each stage puts its layers' state in once it has read that chunk, the last one
    // files it
    const bool ck_on = runtime_.o.prompt_cache > 0 && runtime_.o.prompt_cache_every > 0;
    int64_t ck_next = runtime_.pp_next_check;
    std::map<int64_t, ConvCheckpoint> ck_main;
    runtime_.sp.on_stage_chunk = [&](int64_t done, std::string& ce) -> bool {
        main_done = done;
        if (!ck_on || done < ck_next) return true;
        ck_next = done + runtime_.o.prompt_cache_every;
        ConvCheckpoint part;   // (its stream is synchronized)
        part.ids.assign(runtime_.cur.begin(), runtime_.cur.begin() + done);
        if (!checkpoint_save(part, *runtime_.rsp, runtime_.g)) {
            ce = "saving a conversation checkpoint failed";
            return false;
        }
        std::lock_guard<std::mutex> lk(relay_mu);
        ck_main[done] = std::move(part);
        return true;
    };
    for (PromptStage* stp : run) {
        if (stp == &last) continue;
        stp->sp.on_stage_chunk = [&, stp](int64_t done, std::string& ce) -> bool {
            if (!ck_on) return true;
            std::lock_guard<std::mutex> lk(relay_mu);
            const auto it = ck_main.find(done);
            std::string se;
            if (it != ck_main.end() &&
                !basalt::core::conversation_checkpoint_save_layers(it->second, stp->ss, runtime_.g, stp->l0, stp->l1, se)) {
                ce = "saving a conversation checkpoint failed: " + se;
                return false;
            }
            return true;
        };
    }
    // every stage done with its chunks before the kept one at `c_end`: their state to the main session, and
    // the slots' experts back on a thread while the main card reads the kept chunk
    auto state_back = [&](int64_t c_end, std::string& ce) -> bool {
        // Refill starts in this callback, before the stage run returns.
        if (!runtime_.close_vision_tail(ce)) {
            basalt::log::error("basalt serve", "%s", ce.c_str());
            std::exit(1);
        }
        const auto tc = Clock::now();
        bool ok = true;
        for (PromptStage* st : run)
            ok = ok && basalt::core::session_copy_state(*runtime_.rsp, st->ss, runtime_.g, st->l0, st->l1, a0, c_end,
                                                        [&](void* d, const void* sr, uint64_t nb) {
                                                            return st->state_back.copy(d, sr, nb, ce);
                                                        }, ce);
        for (PromptStage* st : run) ok = ok && st->state_back.wait(ce);
        if (!ok) {
            if (ce.empty()) ce = "prompt split: a stage's state back to the main session";
            return false;
        }
        given = true;
        given_at = c_end;
        ms_copy = std::chrono::duration<double, std::milli>(Clock::now() - tc).count();
        refill_issue = std::async(std::launch::async, [&] {
            bool issued = true;
            for (PromptStage* st : run) issued = st->tier->give_back(st->lent, refill_err) && issued;
            return issued;
        });
        return true;
    };
    last.sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& ce) -> bool {
        const int64_t HCN = runtime_.g.hc * runtime_.g.n_embd;
        const int64_t r0 = std::max<int64_t>(0, first_cell - p0);
        for (int64_t r = r0; r < T && p0 < in_place_from;) {
            if (aside_at >= aside_rows) {
                if (++aside_buf > 1) {
                    ce = "prompt split: the draft layer's rows do not fit the hand-off buffers";
                    return false;
                }
                aside_at = 0;
                continue;
            }
            float* h = before_last.hand_buffer(aside_buf);
            const int64_t nr = std::min(aside_rows - aside_at, T - r);
            float* dst = h == nullptr ? nullptr : h + (size_t) (chunk + aside_at) * HCN;
            if (!relay_copy(dst, R_rows + (size_t) r * HCN, nr * HCN, ce)) {
                ce = "prompt split: reading the draft layer's rows back failed";
                return false;
            }
            std::lock_guard<std::mutex> lk(relay_mu);
            relays.push_back({p0 + r, nr, dst});
            aside_at += nr;
            r += nr;
        }
        if (r0 < T && p0 >= in_place_from) {
            float* h = const_cast<float*>(last.sp.hand_in());
            if (!relay_copy(h ? h + (size_t) r0 * HCN : nullptr, R_rows + (size_t) r0 * HCN, (T - r0) * HCN, ce)) {
                ce = "prompt split: reading the draft layer's rows back failed";
                return false;
            }
            std::lock_guard<std::mutex> lk(relay_mu);
            relays.push_back({p0 + r0, T - r0, h + (size_t) r0 * HCN});
        }
        std::vector<ConvCheckpoint> whole(1);
        {
            std::lock_guard<std::mutex> lk(relay_mu);
            const auto it = ck_main.find(p0 + T);
            if (it != ck_main.end()) {
                whole[0] = std::move(it->second);
                ck_main.erase(it);
            }
        }
        std::string se;
        if (!whole[0].ids.empty() &&
            (!basalt::core::conversation_checkpoint_save_layers(whole[0], last.ss, runtime_.g, last.l0, last.l1, se) ||
             !runtime_.checkpoint_at(p0 + T, &whole))) {
            ce = "saving a conversation checkpoint failed" + (se.empty() ? runtime_.ckpt_why : ": " + se);
            return false;
        }
        // its last chunk before the kept one: every stage is done with its own
        if (p0 + T >= c_cut.load() && !state_back(p0 + T, ce)) return false;
        const int64_t done = p0 + T;
        last_end.store(done);
        runtime_.pp_reached = done;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - runtime_.pp_t0).count();
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) runtime_.pp_total, ms,
                    ms > 0.0 ? 1000.0 * (double) (done - runtime_.pp_from) / ms : 0.0);
        basalt::core::progress_at("reading the prompt (batched), done up to token", done);
        basalt::core::progress_beat();
        std::fflush(stdout);
        return true;
    };
    if (vision_arena) runtime_.offer_vision_tail(vision_arena, vision_bytes);
    bool ok = runtime_.sp.run(ids.data() + a0, n_tok, a0, e);   // (waits for every stage's chunks)
    // An early stop can bypass state_back even if no expert was resident in the loan.
    std::string vision_error;
    if (!runtime_.close_vision_tail(vision_error)) {
        basalt::log::error("basalt serve", "%s", vision_error.c_str());
        std::exit(1);
    }
    runtime_.sp.keep_here = nullptr;
    // the kept chunk [c_end, b_end): the last one, or the one a read that gives way ends with (the stages may
    // have been done before the cut)
    const int64_t c_end = c_cut.load();
    const int64_t b_end = yield_q >= 0 ? yield_q : b0;
    if (ok && !given && yield_q >= 0) ok = state_back(c_end, e);
    // the stages' state must come back exactly at the main card's kept chunk: anything else is an error
    if (ok && (!given || (yield_q < 0 && given_at != c_end))) {
        ok = false;
        e = "prompt split: the stages' state did not come back at the kept chunk (planned at " +
            std::to_string(c_end) + ", the stages ended at " + std::to_string(last_end.load()) +
            (given ? ", it came back at " + std::to_string(given_at) : std::string(", it never came back")) +
            "; " + std::to_string(n_tok) + " tokens in chunks of " + std::to_string(chunk) + ", head " +
            std::to_string(head) + ")";
    }
    // KV streaming: the stages wrote their layers' host copies, so a block resident before may be stale
    if (given && runtime_.g.n_qsa_layers() > 0 && runtime_.rsp->qsa_states[runtime_.rsp->qsa_primary()].kv_mode == 1) {
        const basalt::core::OnDevice on0(0);
        const int64_t I = std::max<int64_t>(runtime_.g.qsa_interval, 1);
        for (PromptStage* st : run)
            for (int64_t q = st->l0 / I; q < st->l1 / I; ++q) {
                const auto& state = runtime_.rsp->qsa_states[q];
                if (state.shared_residency) {
                    const int64_t page = basalt::kernels::qsa_real_shapes().page_size;
                    const int64_t first = a0 / page;
                    const int64_t end = (c_end + page - 1) / page;
                    if (end > first)
                        basalt::kernels::kv_stream_invalidate(state.map, state.host.fp16_host_pages + first,
                                                             end - first, nullptr);
                } else {
                    basalt::kernels::kv_stream_reset(state.map, nullptr);
                }
            }
        if (cudaStreamSynchronize(nullptr) != cudaSuccess && ok) {
            ok = false;
            e = "prompt split: resetting the streamed K/V's residency";
        }
    }
    runtime_.sp.set_keep_last(false);
    runtime_.sp.on_stage_chunk = nullptr;
    for (PromptStage* st : run) {
        st->sp.on_stage_chunk = nullptr;
        st->sp.on_chunk = nullptr;
        st->sp.set_stage(st->l0, -1, nullptr);   // alone again: its routed entries are taken on their own
    }
    runtime_.pp_next_check = ck_next;   // the last chunk's check, on this card as on one card
    const double ms_stages = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    // the draft layer's K/V of the last stage's rows, through this card's prompt path; then the last chunk's
    // layers [lb, n_layers) here
    const auto td = Clock::now();
    for (const Relay& rl : relays) {
        if (!ok || !runtime_.mtp_on) break;
        std::vector<int32_t> nxt((size_t) rl.n);
        for (int64_t t = 0; t < rl.n; ++t) nxt[(size_t) t] = (int32_t) runtime_.cur[(size_t) (rl.cell0 + t + 1)];
        const bool batched = runtime_.sp.draft_kv(runtime_.mtp, rl.rows, nxt.data(), rl.n, rl.cell0, e, draft_fp16);
        if (!batched && draft_fp16 && e.empty()) {
            float* rows = const_cast<float*>(rl.rows);
            const auto* packed = reinterpret_cast<const uint16_t*>(rows);
            const int64_t count = rl.n * runtime_.g.hc * runtime_.g.n_embd;
            for (int64_t i = count; i-- > 0;)
                rows[i] = basalt::kernels::f32_from_f16(packed[i]);
        }
        ok = e.empty() && (batched || runtime_.mtp.prefill(rl.rows, nxt.data(), rl.n, rl.cell0, e));
    }
    relays.clear();
    const double ms_draft = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
    runtime_.sp.set_stage(lb, -1, nullptr);
    if (ok) {
        runtime_.sp.continue_in_place();
        ok = runtime_.sp.run(ids.data() + c_end, b_end - c_end, c_end, e);
    }
    runtime_.sp.set_stage(0, -1, nullptr);
    runtime_.sp.set_peer_share(true);
    {
        const bool issued = refill_issue.valid() && refill_issue.get();
        for (PromptStage* st : run) {
            const basalt::core::OnDevice on(st->dev);
            std::string ge;
            if (!st->lent.empty() && !(ok && issued && st->tier->give_back_wait(st->lent, ge)))
                give_back_now(*st);   // stopped early: nothing of the stage may still write the slots
        }
    }
    runtime_.split_hist.clear();
    runtime_.split_hist_tokens = 0;
    if (!runtime_.drive.d.usage.empty())
        for (PromptStage* st : run) st->sp.route_hist_take(runtime_.split_hist, runtime_.split_hist_tokens);
    if (ok && yield_q >= 0) {   // (a run that ends early leaves the token window of the PLE hash where it began)
        if (yield_q >= 2) {
            runtime_.rsp->ple_prev[0] = (int32_t) ids[(size_t) yield_q - 2];
            runtime_.rsp->ple_prev[1] = (int32_t) ids[(size_t) yield_q - 1];
        }
        std::string ye;
        const auto ty = Clock::now();
        if (!park_read(yield_q, yield_ys, ye)) {
            e = "BYIELD: parking the prompt read failed: " + ye;
            return false;
        }
        basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d holds it "
                             "(copied in %.1f ms)", (long long) yield_q, (long long) n, yield_ys,
                          std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
        e = "yield";
        return false;
    }
    if (!ok) return false;
    const basalt::prefill::PrefillStats& sm1 = runtime_.sp.stats();
    std::string where, cards;
    for (size_t k = 0; k < run.size(); ++k) {
        const PromptStage& st = *run[k];
        const basalt::prefill::PrefillStats& st1 = st.sp.stats();
        size_t pfree = 0, ptotal = 0;
        {
            const basalt::core::OnDevice on(st.dev);
            if (cudaMemGetInfo(&pfree, &ptotal) != cudaSuccess) (void) cudaGetLastError();
        }
        char b[256];
        std::snprintf(b, sizeof b, "%s%lld on CUDA%d, layers %lld-%lld", k ? ", " : "",
                      (long long) (st1.chunks - st0[k].chunks), st.dev, (long long) st.l0, (long long) st.l1 - 1);
        where += b;
        std::snprintf(b, sizeof b, "CUDA%d %lld lent slots, its stage %.0f ms, %lld experts streamed, %zu MiB free "
                      "after; ", st.dev, (long long) st.lent_slots, st1.ms_total - st0[k].ms_total,
                      (long long) (st1.experts_streamed - st0[k].experts_streamed), pfree >> 20);
        cards += b;
    }
    basalt::log::info("basalt serve", "prompt split: %lld tokens in chunks of %lld (%s), %.0f ms (setup %.1f, stages "
                         "%.0f, state back %.0f, draft rows %.0f); %sCUDA0 %.0f ms, %lld streamed",
                      (long long) n_tok, (long long) chunk, where.c_str(),
                      std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), ms_setup, ms_stages,
                      ms_copy, ms_draft, cards.c_str(), sm1.ms_total - sm0.ms_total,
                      (long long) (sm1.experts_streamed - sm0.experts_streamed));
    return true;
}

}  // namespace basalt::models::qwen4exp::serve
