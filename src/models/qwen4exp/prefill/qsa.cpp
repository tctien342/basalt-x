#include "run.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/models/qwen4exp/kernels/native_qsa_indexer.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/kv_stream.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/decode/layer.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_prompt_attn.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_select.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"
#include <algorithm>
#include <atomic>
#include <exception>
#include <string>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::qsa(const core::LayerView& v) {
    // ======================= QSA =======================
    const core::QsaState& st = ss.qsa_states[qsa_index];
    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                          *wik = need(v, "indexer.k_proj.weight", err),
                          *wiq = need(v, "indexer.q_proj.weight", err),
                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                          *wiqn = need(v, "indexer.q_norm.weight", err),
                          *wikn = need(v, "indexer.k_norm.weight", err);
    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
    const bool skip_identity = core::exp_flag("qsa_identity_index") != 0;
    // Prefill steps are contiguous. If the last query selects its entire prefix, the earlier ones do too.
    const auto identity_through = [&](int64_t token) {
        const int32_t* step = m.steps_host.data() + token * basalt::kernels::kStepCount;
        return step[basalt::kernels::kStepNKv] <= step[basalt::kernels::kStepWidth];
    };
    const bool identity = skip_identity && identity_through(T - 1);
    const bool staged = st.kv_mode == 1;
    const int fmt = staged ? core::qsa_kv_format(st) : basalt::kernels::kKvF16;
    const int64_t cached_blocks = (p0 + s.page_size - 1) / s.page_size;
    const bool mapped_staging = fmt == basalt::kernels::kKvF16 || fmt == basalt::kernels::kKvInt8;
    const bool selective = staged && mapped_staging && run.first_stage && run.single_chunk &&
        (p0 >= 65536 || (p0 >= 32768 && T <= 640) || (p0 >= 16384 && T <= 256)) &&
        T >= 64 && T <= 2560 && cached_blocks <= m.max_blocks &&
        s.n_head_kv == 2 && s.page_size == 4 && s.head_dim == 256 &&
        st.host.present() && st.page_table &&
        core::exp_flag("qsa_selective_stage") != 0;
    const bool overlap = !selective && staged && cached_blocks > 0 && mapped_staging && m.kv_copy_done;
    if (overlap) {
        // Previous attention and host KV writes finish before the copy; this layer's projections can run beside it.
        if (cudaEventRecord(m.kv_copy_start, m.cs) != cudaSuccess ||
            cudaStreamWaitEvent(m.copy, m.kv_copy_start, 0) != cudaSuccess) {
            err = "prefill: begin KV staging";
            return false;
        }
        basalt::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host, fmt, cached_blocks, s, m.copy);
        if (cudaEventRecord(m.kv_copy_done, m.copy) != cudaSuccess) {
            err = "prefill: record KV staging";
            return false;
        }
        static std::atomic<bool> noted{false};
        if (!noted.exchange(true)) basalt::log::info("basalt", "prefill: overlapping KV staging and QSA projections");
    }
    const bool projected = native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err) &&
                           native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err) &&
                           native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err) &&
                           bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err) &&
                           (identity || bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err));
    const auto join_kv_copy = [&] {
        if (cudaStreamWaitEvent(m.cs, m.kv_copy_done, 0) == cudaSuccess) return true;
        err = "prefill: wait for KV staging";
        return false;
    };
    if (!projected) {
        if (overlap) join_kv_copy();
        return false;
    }
    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
    rope(m.Kc, T, 2, 256, 512, p0, basalt::kernels::rope_scaling(), m.cs);
    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
    if (staged && !overlap && !selective) {
        basalt::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host, fmt, cached_blocks, s, m.cs);
    }
    const auto append_kv = [&] {
        const basalt::kernels::KvHostPools* host_w = &st.host;
        if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
            basalt::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
            kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                      st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, nullptr,   // mode 0: no host mirror
                      staged ? &m.stage : nullptr);
            basalt::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                          nullptr, staged ? &m.stage : nullptr);
        } else {
            if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
                basalt::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                basalt::kernels::fwht_value_inplace_cuda(m.Vc, T * 2, m.cs,
                                                         basalt::kernels::kv_value_rotation_width(st.kv_int8 && !st.kv_q4));
            }
            if (st.kv_q4)
                basalt::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                              host_w, staged ? &m.stage : nullptr);
            else
                kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                          st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                          host_w, staged ? &m.stage : nullptr);
        }
    };
    if (!overlap && !selective) append_kv();
    split_q(m.Qf, m.q, T, m.cs);
    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
    rope(m.q, T, 24, 256, 6144, p0, basalt::kernels::rope_scaling(), m.cs);
    if (st.kv_rot) basalt::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
    if (identity) {
        static std::atomic<bool> noted{false};
        if (!noted.exchange(true)) basalt::log::info("basalt", "prefill: QSA identity skips query projection");
    } else {
        rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
        rope(m.q_idx, T, 4, 128, 512, p0, basalt::kernels::rope_scaling(), m.cs);
    }
    // the indexer appends, token by token; then scores + selection for many queries at once:
    // a query reads completed blocks (final once completed) and `dead` for its own tail block
    const basalt::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
    // the chunk's appends in three launches instead of one per token (the same end state:
    // the queries below read it only after the whole chunk is appended).
    try {
        basalt::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                         EPS, ib, s, st.max_cells,
                                                         basalt::kernels::rope_scaling(), m.cs);
    } catch (const std::exception& e) {
        err = std::string("prefill indexer: ") + e.what();
        if (overlap) join_kv_copy();
        return false;
    }
    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
        const int64_t nb = std::min(m.sel_batch, T - t0);
        const int32_t* steps0 = m.steps_dev + t0 * basalt::kernels::kStepCount;
        // the grid reaches the batch's last query's n_bid (they rise with the position)
        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * basalt::kernels::kStepCount +
                                                                basalt::kernels::kStepNBid)] + 1;
        if (skip_identity && identity_through(t0 + nb - 1)) {
            // Every selector emits ascending IDs before reading any scores for these queries.
            if (!identity && t0 == 0) {
                static std::atomic<bool> noted{false};
                if (!noted.exchange(true)) basalt::log::info("basalt", "prefill: QSA identity skips initial scores");
            }
        } else if (!basalt::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                        steps0, nb, m.max_blocks, s, m.sel_scores, m.cs, active)) {
            basalt::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                              m.max_blocks, s, m.sel_scores, m.cs, active);
        }
        basalt::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                        m.sel_ids + t0 * m.cap, m.cs, active);
    }
    // Selection reads the indexer state, so it can finish before the staged K/V pages are needed.
    if (overlap) {
        if (!join_kv_copy()) return false;
        append_kv();
    }
    const basalt::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                       : core::qsa_attn_pools(st);
    if (selective) {
        // Scores are dead after selection. Stage historical pages before appending a possibly shared tail page.
        if (!basalt::kernels::kv_stage_selected(pools, core::qsa_attn_pools(st), st.host, fmt, s,
                reinterpret_cast<uint32_t*>(m.sel_scores), m.sel_ids, m.steps_dev,
                (int) T, (int) m.cap, (int) p0, m.cs)) {
            err = "prefill: stage selected KV pages";
            return false;
        }
        append_kv();
        static std::atomic<bool> noted{false};
        if (!noted.exchange(true))
            basalt::log::info("basalt prefill", "selective KV staging for cached QSA");
    }
    // the whole chunk on tensor cores, one block per (query, KV head), FP32-level
    // accuracy but not bitwise (qsa_prompt_attn.hpp). Q4_0 KV (or a shape it does not take): the
    // decode kernel, 32 queries at a time (K8V4 runs the tensor kernel's mode 3: INT8 K,
    // V dequantized from its q4_0 blocks to fp16 at gather)
    if (!basalt::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                            m.attn, T, m.cs))
        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
            const int64_t nb = std::min(m.attn_batch, T - t0);
            basalt::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                   m.steps_dev + t0 * basalt::kernels::kStepCount, m.cap,
                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
        }
    if (st.kv_rot || st.kv_hybrid)
        basalt::kernels::fwht_value_inplace_cuda(m.attn, T * 24, m.cs,
                                                 basalt::kernels::kv_value_rotation_width(st.kv_rot && st.kv_int8 && !st.kv_q4));
    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
    ++qsa_index;
    return true;
}

}  // namespace basalt::prefill
