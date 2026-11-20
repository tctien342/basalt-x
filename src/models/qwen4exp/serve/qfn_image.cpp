// src/models/qwen4exp/serve/qfn_image.cpp - a request's image positions and the quality gate's chunk.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "basalt/vision/mrope_positions.hpp"

namespace basalt::models::qwen4exp::serve {

bool QfnRequest::parse_begin() { return !runtime_.mtp_on || runtime_.mtp.use_main(runtime_.err); }
Flow QfnRequest::bind_context_profile() {
    if (!runtime_.rope_context_profiles) return Request::bind_context_profile();
    using basalt::kernels::RopeContext;
    rope_scope_.emplace(req_rope_profile == 1 ? RopeContext::YaRN2 : RopeContext::Native);
    return Flow::proceed;
}

int64_t QfnRequest::limit_context(int64_t context) const {
    if (!runtime_.rope_context_profiles) return context;
    return std::min<int64_t>(context, req_rope_profile == 1 ? 524288 : 262144);
}

Flow QfnRequest::bind_atomic_lora() {
    auto& r = runtime_;
    const unsigned mode = request_adapter();
    const unsigned key = r.cache_key(mode, basalt::kernels::rope_context());
    if (!r.cache_configured(key)) return Request::bind_atomic_lora();
    // The frozen image is not visible to restore until the worker finishes.
    // Resolve it before searching the outgoing namespace for a prefix match.
    if (r.frozen_recovery.valid() && key == r.frozen_recovery_cache && !r.settle_shared_slot(0, false)) {
        std::printf("ERR cold shared snapshot: %s\n", r.err.c_str());
        return Flow::error;
    }
    const bool profile_switch = (key >= QfnRuntime::adapter_count) != (r.active_cache >= QfnRuntime::adapter_count);
    bool deferred = false;
    if (profile_switch && !geni && r.admit_slot > 0 && basalt::core::exp_flag("cold_shared_recovery")) {
        const int64_t ctx = limit_context(r.slot_context(r.admit_slot));
        if (n > 0 && ctx >= 8 && n <= ctx - 8 && r.admit_max_new <= ctx - 8 - n) {
            if (mode == r.active_adapter) {
                deferred = r.defer_profile_park(r.admit_slot, n + r.admit_max_new + 8);
            } else if (r.shared_pages && r.slot0_solo() && !r.pipe_on && !r.piped && r.stages.empty() &&
                       !r.frozen_recovery.valid() && r.rsp == &r.ss && r.live_ok && !r.live.empty() &&
                       r.live_imgs.empty() && !r.bs[0].active && !r.bs[0].cached && !r.bs[0].partial &&
                       r.cached_context_key(0) == r.active_cache) {
                auto& source = *r.shared_sequences[0];
                const auto& destination = *r.shared_sequences[(size_t) r.admit_slot];
                const auto page = basalt::kernels::qsa_real_shapes().page_size;
                const auto source_pages = (uint32_t) (1 + (r.live.size() - 1) / page);
                const auto needed = (uint64_t) (1 + (n + r.admit_max_new + 7) / page);
                if (needed <= (uint64_t) r.shared_pages->available() + destination.lease.pages().size() &&
                    basalt::core::SharedKvTable::covers_prefix(source.lease, source.stages, source_pages)) {
                    // Publish the old solo checkpoint chain in its existing lease. Preserve its
                    // draft store under the outgoing context before selecting the incoming adapter.
                    const basalt::kernels::RopeContextScope outgoing(source.rope_context);
                    r.ckd_finish();
                    if (!r.ver.wait_commit(r.err) || (r.thinking_ver && !r.thinking_ver->wait_commit(r.err))) {
                        std::printf("ERR pending commit before slot handoff: %s\n", r.err.c_str());
                        return Flow::error;
                    }
                    if (!r.batch_spec || !r.lora_main || r.mtp.set_store_adapter_scale(0,
                            (r.active_adapter & 1u) ? QfnRuntime::atomic_lora_scale : 0.0f, r.err)) {
                        slot_draft_out();
                        if (!r.batch_spec || r.solo_draft_in0) {
                            auto& slot = r.bs[0];
                            slot = {};
                            slot.p = (int64_t) r.live.size();
                            slot.ids = std::move(r.live);
                            slot.checks = std::move(r.checks);
                            slot.cvec = r.cvec_cached;
                            slot.idle_busy = r.solo_idle_busy;
                            slot.cached = true;
                            r.solo_draft_in0 = false;
                            deferred = true;
                        }
                    }
                    if (!deferred) r.err.clear();
                }
            }
        }
    }
    if (r.lora_main && r.mtp_on && !r.mtp.set_store_adapter_scale(r.mtp.main_store(),
            req_atomic_lora ? QfnRuntime::atomic_lora_scale : 0.0f, r.err)) {
        std::printf("ERR AtomicChat draft scale: %s\n", r.err.c_str());
        return Flow::error;
    }
    if (key == r.active_cache) return Flow::proceed;
    if (!deferred && !r.park_current(0, 0, profile_switch)) {
        std::printf("ERR could not park the outgoing request session\n");
        return Flow::error;
    }
    if (mode != r.active_adapter) {
        const float scale = req_atomic_lora ? QfnRuntime::atomic_lora_scale : 0.0f;
        try {
            const auto sync = [](int device) {
                const basalt::core::OnDevice on(device);
                const auto status = cudaDeviceSynchronize();
                if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            };
            sync(0);
            if (r.peer.valid() && (r.lora_peer || r.swift_peer)) sync(r.peer.device());
            if (r.lora_main) r.lora_main->set_scale(scale);
            if (r.lora_peer) r.lora_peer->set_scale(scale);
            const float swift = req_swift_lora ? QfnRuntime::swift_lora_scale : 0.0f;
            if (r.swift_main) r.swift_main->set_scale(swift);
            if (r.swift_peer && r.peer.valid()) r.swift_peer->set_scale(swift);
        } catch (const std::exception& error) {
            std::printf("ERR LoRA scale switch: %s\n", error.what());
            return Flow::error;
        }
        r.ver.set_solo_swift(req_swift_lora);
        if (r.thinking_ver) r.thinking_ver->set_solo_swift(req_swift_lora);
        r.sp.bind_swift_lora(req_swift_lora ? r.swift_main : nullptr);
        r.sp.set_cpu_lora(r.lora_host.get(), scale);
        for (auto& stage : r.pstages) {
            stage->sp.bind_swift_lora(req_swift_lora ? r.swift_peer : nullptr);
            stage->sp.set_cpu_lora(r.lora_host.get(), scale);
        }
        r.drive.d.cpu_lora = r.lora_host.get();
        r.drive.d.cpu_lora_scale = scale;
    }
    r.conversations->drop_reuse_unless(nullptr);
    r.conversations.swap(r.parked_conversations[r.active_cache]);
    r.conversations.swap(r.parked_conversations[key]);
    r.live.clear();
    r.live_imgs.clear();
    r.checks.clear();
    r.live_ok = false;
    r.active_adapter = mode;
    r.active_cache = key;
    return Flow::proceed;
}

float QfnRequest::default_relax() const { return runtime_.o.relaxed_drafting; }

Flow QfnRequest::check_prepared_capacity(int64_t cells) {
    const int slot = runtime_.solo_slot >= 0 ? runtime_.solo_slot : runtime_.admit_slot;
    const bool own_slot = runtime_.slot0_solo() && slot > 0;
    const int64_t page_cells = basalt::kernels::qsa_real_shapes().page_size;
    const auto pages = size_t(1 + (cells - 1) / page_cells);
    if (runtime_.shared_pages) {
        if (pages > runtime_.shared_pages->capacity()) {
            std::printf("ERR shared KV: request exceeds pool capacity\n");
            std::fflush(stdout);
            return Flow::next;
        }
        if (const auto result = runtime_.publish_shared_request(own_slot ? slot : 0, static_cast<uint32_t>(pages),
                                                               request_adapter());
            result != Flow::proceed) return result;
    }
    const auto covered = [pages](const basalt::core::SessionState& session) {
        for (int64_t i = 0; i < session.qsa_alloc; ++i) {
            const auto& state = session.qsa_states[session.qsa_ord0 + i];
            if (state.shared_residency &&
                (!state.host.fp16_host_pages || !state.host_page_ids || pages > state.host_page_count))
                return false;
        }
        return true;
    };
    const auto all_covered = [&] {
        if (own_slot) {
            for (const auto& stage : runtime_.bslot_ss)
                if (!covered(*stage[(size_t) slot])) return false;
            return true;
        }
        if (!covered(runtime_.ss)) return false;
        for (const auto& stage : runtime_.stages)
            if (!covered(stage->ss)) return false;
        return true;
    };
    if (all_covered()) return Flow::proceed;
    std::printf("ERR shared KV: request pages were not published before generation\n");
    std::fflush(stdout);
    return Flow::next;
}

Flow QfnRequest::prepare_image() {
    if (const Flow flow = begin_quality_gate(runtime_.ppl_base, runtime_.ppl_out, runtime_.S); flow != Flow::proceed) return flow;
    if (geni || !runtime_.mrope_identity) {
        // positions for every cell this request can reach; the identity again for a text request
        std::string ve;
        runtime_.row_ptr.assign((size_t) n, nullptr);
        const int64_t cells = (int64_t) runtime_.mrope_host.size() / 3;
        auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
            runtime_.mrope_host[(size_t) c * 3] = (int32_t) t;
            runtime_.mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
            runtime_.mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
        };
        if (!geni) {
            for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
        } else {
            struct Img { int64_t n, nx, ny; size_t off; };
            std::vector<Img> imgs;
            runtime_.img_rows.clear();
            std::FILE* f = std::fopen(emb_path.c_str(), "rb");
            if (!f) ve = "cannot open " + emb_path;
            while (f && ve.empty()) {
                int32_t hdr[5];
                const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                if (got == 0) break;
                if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                    (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) runtime_.g.n_embd) {
                    ve = "bad embeddings file (expected basalt-vision records of width " +
                         std::to_string((long long) runtime_.g.n_embd) + ")";
                    break;
                }
                const size_t off = runtime_.img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                runtime_.img_rows.resize(off + cnt);
                if (std::fread(runtime_.img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                imgs.push_back({hdr[1], hdr[2], hdr[3], off});
            }
            if (f) std::fclose(f);
            // the table holds one entry per cell: a longer prompt is refused before any is written
            if (ve.empty() && (n > cells || n > runtime_.o.max_context))
                ve = "prompt (" + std::to_string((long long) n) + " tokens) exceeds the context (" +
                     std::to_string((long long) runtime_.o.max_context) + ")";
            // each run of a picture's pad or a video group's takes the next record, in order
            std::vector<int64_t> pads = {runtime_.o.image_pad_id}, runs;
            if (runtime_.o.video_pad_id >= 0) pads.push_back(runtime_.o.video_pad_id);
            std::vector<basalt::vision::MediaGrid> grids;
            for (const Img& im : imgs) grids.push_back({im.n, im.nx, im.ny});
            if (ve.empty()) ve = basalt::vision::mrope_positions(ids.data(), n, pads, grids, cells, runtime_.mrope_host.data(), runs);
            for (size_t k = 0; ve.empty() && k < runs.size(); ++k) {
                const Img& im = imgs[k];
                const int64_t i = runs[k];
                {   // what the conversation cache compares: a picture is its pad, its grid and its embeddings
                    const int64_t grid[4] = {ids[(size_t) i], im.n, im.nx, im.ny};
                    uint64_t h = fnv1a(grid, sizeof grid);
                    h = fnv1a(runtime_.img_rows.data() + im.off, (size_t) im.n * (size_t) runtime_.g.n_embd * sizeof(float), h);
                    runtime_.req_imgs.push_back({i, h});
                }
                for (int64_t j = 0; j < im.n; ++j)
                    runtime_.row_ptr[(size_t) (i + j)] = runtime_.img_rows.data() + im.off + (size_t) j * (size_t) runtime_.g.n_embd;
            }
        }
        cudaDeviceSynchronize();
        // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
        auto upload_mrope = [&]() -> bool {
            bool ok = cudaMemcpy(runtime_.d_mrope, runtime_.mrope_host.data(), runtime_.mrope_host.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice) == cudaSuccess;
            for (auto& st : runtime_.stages) {
                const basalt::core::OnDevice on(st->dev);
                cudaDeviceSynchronize();
                ok = ok && cudaMemcpy(st->mrope, runtime_.mrope_host.data(), runtime_.mrope_host.size() * sizeof(int32_t),
                                      cudaMemcpyHostToDevice) == cudaSuccess;
            }
            return ok;
        };
        if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
        if (!ve.empty()) {
            // leave the table as the identity so the next text request is untouched
            for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
            upload_mrope();
            runtime_.mrope_identity = true;
            std::printf("ERR %s\n", ve.c_str());
            std::fflush(stdout);
            return Flow::next;
        }
        runtime_.mrope_identity = !geni;
    }
    runtime_.sp.embd_rows = geni ? runtime_.row_ptr.data() : nullptr;
    return Flow::proceed;
}

}  // namespace basalt::models::qwen4exp::serve
