#include "basalt/models/qwen4exp/decode/conversation_snapshot.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "conversation_checked.hpp"
#include "basalt/core/diag.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <limits>

namespace basalt::core {
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled, value_data, value_scales;
};

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = basalt::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

bool layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index, Layout& l, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_mode != 0 || st.kv_q4 || st.kv_int8)) {
        error = "conversation snapshot: hybrid K8V4 requires an identity layout and distinct format flags";
        return false;
    }
    const auto s = basalt::kernels::qsa_real_shapes();
    const bool int8_keys = st.kv_int8 || st.kv_hybrid;
    if (g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim > INT32_MAX || g.idx_key_dim <= 0 ||
        (st.kv_q4 && g.head_dim % 32) || (int8_keys && !st.kv_q4 && g.head_dim % 64)) {
        error = "conversation snapshot: invalid K/V geometry";
        return false;
    }
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    const size_t per = st.kv_q4 ? (size_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                              : (size_t) g.head_dim * (int8_keys ? 1 : 2);
    // Include the moving spare row, not only completed blocks. The checkpoint
    // restore reconstructs that row when rewinding to an earlier prefix.
    const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;
    // Format 3 identifies snapshot K8V4 only; never pass it to the block movers.  Rotated INT8 K/V
    // (KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int width = basalt::kernels::kv_value_rotation_width(st.kv_rot && int8_keys && !st.kv_q4 && !st.kv_hybrid);
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid
        ? (width == 64 ? 32 : width == 128 ? 48 : 16) : 0;
    l = {(st.kv_hybrid ? 3 : qsa_kv_format(st)) + rotated, cells, pooled, s.page_size, 0, 0, 0, 0, 0};
    using conversation_detail::product;
    if (!product(l.data, {(uint64_t) cells, (uint64_t) g.n_head_kv, per}) ||
        !product(l.scales, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                           int8_keys && !st.kv_q4 ? (uint64_t) (g.head_dim / 64) * 2 : 0}) ||
        !product(l.pooled, {(uint64_t) pooled, (uint64_t) g.idx_key_dim, sizeof(float)})) {
        error = "conversation snapshot: K/V byte count overflow";
        return false;
    }
    l.value_data = l.data;
    l.value_scales = l.scales;
    if (st.kv_hybrid) {
        if (!product(l.value_data, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                   (uint64_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim)})) {
            error = "conversation snapshot: hybrid V byte count overflow";
            return false;
        }
        l.value_scales = 0;
    }
    return true;
}

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid) return {st.k_q, st.v_q4, st.k_scale, nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || l.cells / l.page_size > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && st.n_slots < st.n_pages) ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if ((st.shared_residency ? m.n_blocks <= 0 || (size_t) m.n_blocks != st.host_pool_pages : m.n_blocks != st.n_pages) || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            error = "conversation snapshot: invalid streaming map";
            return false;
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                    : !st.k_pool || !st.v_pool))) {
        error = "conversation snapshot: invalid draft ring";
        return false;
    }
    return true;
}

// An elastic state backs only a prefix of its cells; a copy past it would fault inside the driver.
bool backed(const QsaState& st, const Layout& l, std::string& error) {
    if (!st.kv_arena || st.kv_arena->cells() >= l.cells) return true;
    error = "conversation snapshot: " + std::to_string((long long) l.cells) + " cells to copy, " +
            std::to_string((long long) st.kv_arena->cells()) + " backed";
    return false;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    const cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDefault);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}
void set_image_layout(ConversationKv& image, const ModelGeometry& g, const Layout& l) {
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
}

// Checkpoint payloads stay in logical page order. Each side can have its own backing.
bool check_host_pages(const QsaState& st, const Layout& l, std::string& error) {
    if (!st.host.fp16_host_pages) return true;
    const size_t count = size_t(l.cells / l.page_size);
    const auto buffers = pools(st);
    if (!st.kv_mode || st.kv_q4 || st.kv_hybrid || !buffers[0] || !buffers[1] ||
        (st.kv_int8 && (!buffers[2] || !buffers[3])) ||
        !st.host_page_ids || count > st.host_page_count) {
        error = "conversation snapshot: missing host K/V page mapping";
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (st.host_page_ids[i] >= st.host_pool_pages) {
            error = "conversation snapshot: host page outside shared pool";
            return false;
        }
    }
    return true;
}

template<class Copy>
bool visit_host_runs(const uint32_t* src, const uint32_t* dst, size_t page_bytes,
                     size_t at, size_t bytes, Copy copy) {
    while (bytes) {
        const size_t page = at / page_bytes, in_page = at % page_bytes;
        const size_t s = src ? src[page] : page;
        const size_t d = dst ? dst[page] : page;
        size_t n = std::min(bytes, page_bytes - in_page);
        for (size_t delta = 1; n < bytes; ++delta) {
            if ((src && src[page + delta] != s + delta) ||
                (dst && dst[page + delta] != d + delta)) break;
            n += std::min(bytes - n, page_bytes);
        }
        if (!copy(s * page_bytes + in_page, d * page_bytes + in_page, at, n)) return false;
        at += n;
        bytes -= n;
    }
    return true;
}
} // namespace

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes, const ConversationCaptureStop* stop) {
    if (conversation_capture_interrupted(stop, error)) return false;
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error) || !backed(st, l, error)) return false;
    if (!check_host_pages(st, l, error)) return false;
    if (unchanged_tokens < 0 || unchanged_tokens > upto || unchanged_tokens > image.cells) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    const int64_t whole_cells = (unchanged_tokens / l.page_size) * l.page_size;
    set_image_layout(image, g, l);
    const auto src = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < dst.size(); ++i) {
        if (conversation_capture_interrupted(stop, error)) return false;
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged_tokens / basalt::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        const uint8_t* host = nullptr;
        if (upto >= 32768 && st.kv_mode != 0 && i < 4 && sizes[i] > keep && src[i] &&
            exp_flag("park_parallel_copy") != 0) {
            // The caller has fenced this layer's writes. Resolve the CPU alias of its authoritative host pool.
            cudaPointerAttributes attr{};
            if (cudaPointerGetAttributes(&attr, src[i]) == cudaSuccess && attr.type == cudaMemoryTypeHost)
                host = static_cast<const uint8_t*>(attr.hostPointer);
        }
        dst[i]->resize_for_overwrite(sizes[i]);   // every new byte is copied in below
        if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                if (conversation_capture_interrupted(stop, error)) return false;
                if (i < 4 && n && st.host.fp16_host_pages)
                    return visit_host_runs(st.host_page_ids, nullptr, sizes[i] / size_t(l.cells / l.page_size), at, n,
                        [&](size_t from, size_t, size_t logical, size_t run) {
                            if (conversation_capture_interrupted(stop, error)) return false;
                            if (host) {
                                std::memcpy(p + logical - at, host + from, run);
                                return true;
                            }
                            return transfer(p + logical - at, static_cast<const uint8_t*>(src[i]) + from, run, error);
                        });
                if (host) {
                    std::memcpy(p, host + at, n);
                    return true;
                }
                return transfer(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, n, error);
            })) return false;
        if (reused_bytes) *reused_bytes += keep;
    }
    return true;
}

bool conversation_kv_seal_shared_draft(ConversationKv& image, ConversationSharedDraftTail& tail,
                                       const QsaState& st, const ModelGeometry& g, int64_t upto,
                                       int64_t prefix, std::string& error) {
    if (prefix < 0 || prefix >= upto || st.kv_mode != 2 || st.host.fp16_host_pages)
        return conversation_kv_save(image, st, g, upto, false, error);
    Layout l{};
    if (!layout(st, g, upto, false, l, error) || !valid(st, l, upto, error) || !backed(st, l, error)) return false;
    const int64_t first = ((prefix + l.page_size - 1) / l.page_size) * l.page_size;
    if (first >= l.cells) return conversation_kv_save(image, st, g, upto, false, error);
    const auto src = pools(st);
    const std::array<size_t, 4> sizes = {l.data, l.value_data, l.scales, l.value_scales};
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (!sizes[i]) continue;
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, src[i]) != cudaSuccess ||
            attr.type != cudaMemoryTypeHost || !attr.hostPointer)
            return conversation_kv_save(image, st, g, upto, false, error);
        tail.host[i] = static_cast<const uint8_t*>(attr.hostPointer);
    }
    set_image_layout(image, g, l);
    ConversationBuffer* target[4] = {&image.k, &image.v, &image.k_scale, &image.v_scale};
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (!sizes[i]) continue;
        target[i]->resize_for_overwrite(sizes[i]);
        if (!target[i]->visit(0, sizes[i] / l.cells * first, [&](uint8_t* dst, size_t count, size_t offset) {
                std::memcpy(dst, tail.host[i] + offset, count);
                return true;
            })) {
            error = "conversation snapshot: cannot seal draft prefix";
            return false;
        }
    }
    tail.first_cell = first;
    tail.end_cell = l.cells;
    return true;
}

bool conversation_snapshot_read_shared_draft(ConversationSharedSnapshot& job, std::string& error) {
    auto& tail = job.draft_tail;
    if (!tail.end_cell) return true;
    if (job.image.kv.empty() || tail.first_cell < 0 || tail.first_cell >= tail.end_cell) {
        error = "conversation snapshot: invalid frozen draft interval";
        return false;
    }
    auto& draft = job.image.kv.back();
    ConversationBuffer* target[4] = {&draft.k, &draft.v, &draft.k_scale, &draft.v_scale};
    for (size_t i = 0; i < 4; ++i) {
        const size_t bytes = target[i]->size();
        if (!bytes) continue;
        if (!tail.host[i] || bytes % tail.end_cell) {
            error = "conversation snapshot: invalid frozen draft backing";
            return false;
        }
        const size_t prefix = bytes / tail.end_cell * tail.first_cell;
        if (!target[i]->visit(prefix, bytes - prefix, [&](uint8_t* dst, size_t count, size_t offset) {
                std::memcpy(dst, tail.host[i] + offset, count);
                return true;
            })) {
            error = "conversation snapshot: cannot copy draft tail";
            return false;
        }
    }
    tail = {};
    return true;
}

bool conversation_kv_seal_shared(ConversationKv& image, ConversationSharedKvSource& copy,
                                 const QsaState& st, const ModelGeometry& g, int64_t upto,
                                 std::span<const uint32_t> pages, std::string& error, bool frozen_index) {
    if (!st.shared_residency || st.kv_mode != 1 || !st.kv_int8 || st.kv_q4 || st.kv_hybrid ||
        !st.host.fp16_host_pages || !st.host_page_ids) {
        error = "conversation snapshot: target is not shared mapped INT8 KV";
        return false;
    }
    Layout l{};
    if (!layout(st, g, upto, true, l, error) || !valid(st, l, upto, error) ||
        !backed(st, l, error) || !check_host_pages(st, l, error)) return false;
    if (pages.empty() || pages.size() != size_t(l.cells / l.page_size) ||
        !std::equal(pages.begin(), pages.end(), st.host_page_ids)) {
        error = "conversation snapshot: target layers disagree on the frozen page map";
        return false;
    }
    ConversationKv sealed;
    set_image_layout(sealed, g, l);
    ConversationSharedKvSource source;
    source.logical_bytes = {l.data, l.value_data, l.scales, l.value_scales};
    const auto src = pools(st);
    for (size_t i = 0; i < source.host.size(); ++i) {
        cudaPointerAttributes attr{};
        const auto status = cudaPointerGetAttributes(&attr, src[i]);
        if (status != cudaSuccess || attr.type != cudaMemoryTypeHost || !attr.hostPointer) {
            error = "conversation snapshot: target is not backed by a readable host alias";
            return false;
        }
        source.host[i] = static_cast<const uint8_t*>(attr.hostPointer);
    }
    if (frozen_index) {
        source.pooled = src[4];
        source.pooled_bytes = l.pooled;
    } else {
        sealed.pooled.resize_for_overwrite(l.pooled);
        if (!sealed.pooled.visit(0, l.pooled, [&](uint8_t* bytes, size_t count, size_t offset) {
                return transfer(bytes, static_cast<const uint8_t*>(src[4]) + offset, count, error);
            })) return false;
    }
    image = std::move(sealed);
    copy = source;
    return true;
}

bool conversation_kv_finish_shared(ConversationKv& image, const ConversationSharedKvSource& copy,
                                   std::span<const uint32_t> pages, std::string& error,
                                   const ConversationCaptureStop* stop) {
    if (pages.empty()) {
        error = "conversation snapshot: empty frozen page map";
        return false;
    }
    const std::array<ConversationBuffer*, 4> target = {&image.k, &image.v, &image.k_scale, &image.v_scale};
    for (size_t i = 0; i < target.size(); ++i) {
        const size_t bytes = copy.logical_bytes[i];
        if (!copy.host[i] || !bytes || bytes % pages.size()) {
            error = "conversation snapshot: invalid frozen host payload";
            return false;
        }
        if (conversation_capture_interrupted(stop, error)) return false;
        target[i]->resize_for_overwrite(bytes);
        if (!target[i]->visit(0, bytes, [&](uint8_t* dst, size_t count, size_t offset) {
                return visit_host_runs(pages.data(), nullptr, bytes / pages.size(), offset, count,
                    [&](size_t physical, size_t, size_t logical, size_t run) {
                        if (conversation_capture_interrupted(stop, error)) return false;
                        std::memcpy(dst + logical - offset, copy.host[i] + physical, run);
                        return true;
                    });
            })) return false;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    return true;
}

bool conversation_detail::trim_kv(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                  int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t, 5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*, 5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (sizes[i] > buffers[i]->size()) {
            error = "conversation snapshot: prefix exceeds the source K/V payload";
            return false;
        }
    }
    for (size_t i = 0; i < buffers.size(); ++i) buffers[i]->resize(sizes[i]);
    image.cells = l.cells;
    image.pooled_rows = l.pooled_rows;
    return true;
}

namespace {
// After the authoritative pools are written: the resident copies refill from them before any attention reads.
bool settle_resident(const QsaState& st, const ModelGeometry& g, int64_t upto, std::string& error) {
    // VRAM slots still contain the outgoing conversation. Resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if (st.shared_residency) {
        const int64_t blocks = (upto + basalt::kernels::qsa_real_shapes().page_size - 1) /
                               basalt::kernels::qsa_real_shapes().page_size;
        basalt::kernels::kv_stream_invalidate(st.map, st.host.fp16_host_pages, blocks, nullptr);
    } else if (st.kv_mode == 1) {
        basalt::kernels::kv_stream_reset(st.map, nullptr);
    }
    if (st.kv_mode == 2 && upto > 0) {
        auto shapes = basalt::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        const int64_t end = (upto + shapes.page_size - 1) / shapes.page_size;
        basalt::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                        std::max<int64_t>(0, end - st.n_slots), end, st.n_slots, shapes, nullptr);
    }
    const auto status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot residency restore: ") + cudaGetErrorString(status);
    return false;
}
}  // namespace

bool conversation_detail::restore_kv(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                     int64_t upto, bool index, bool mapped_host, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !backed(st, l, error) || !check_host_pages(st, l, error)) return false;
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i) {
        uint8_t* host = nullptr;
        if (mapped_host && st.kv_mode != 0 && i < 4 && !src[i]->empty()) {
            cudaPointerAttributes attr{};
            if (cudaPointerGetAttributes(&attr, dst[i]) == cudaSuccess && attr.type == cudaMemoryTypeHost)
                host = static_cast<uint8_t*>(attr.hostPointer);
        }
        if (!src[i]->visit(0, src[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                if (i < 4 && n && st.host.fp16_host_pages)
                    return visit_host_runs(nullptr, st.host_page_ids, src[i]->size() / size_t(l.cells / l.page_size), at, n,
                        [&](size_t, size_t to, size_t logical, size_t run) {
                            if (host) {
                                std::memcpy(host + to, p + logical - at, run);
                                return true;
                            }
                            return transfer(static_cast<uint8_t*>(dst[i]) + to, p + logical - at, run, error);
                        });
                if (host) {
                    std::memcpy(host + at, p, n);
                    return true;
                }
                return transfer(static_cast<uint8_t*>(dst[i]) + at, p, n, error);
            })) return false;
    }
    return settle_resident(st, g, upto, error);
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    return conversation_detail::restore_kv(image, st, g, upto, index, false, error);
}

bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g, int64_t upto, bool index,
                          std::string& error) {
    Layout a{}, b{};
    if (!layout(from, g, upto, index, a, error) || !valid(from, a, upto, error) || !backed(from, a, error) ||
        !layout(to, g, upto, index, b, error) || !valid(to, b, upto, error) || !backed(to, b, error))
        return false;
    if (a.format != b.format || a.cells != b.cells || a.pooled_rows != b.pooled_rows || a.page_size != b.page_size ||
        a.data != b.data || a.scales != b.scales || a.pooled != b.pooled || a.value_data != b.value_data ||
        a.value_scales != b.value_scales) {
        error = "conversation snapshot: the two sessions' K/V layouts differ";
        return false;
    }
    if (!check_host_pages(from, a, error) || !check_host_pages(to, b, error)) return false;
    const auto src = pools(from);
    const auto dst = pools(to);
    const std::array<size_t,5> sizes = {a.data, a.value_data, a.scales, a.value_scales, a.pooled};
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
        const QsaState* src_map = i < 4 && from.host.fp16_host_pages ? &from : nullptr;
        const QsaState* dst_map = i < 4 && to.host.fp16_host_pages ? &to : nullptr;
        if (sizes[i] && (src_map || dst_map)) {
            if (!visit_host_runs(src_map ? src_map->host_page_ids : nullptr,
                                 dst_map ? dst_map->host_page_ids : nullptr,
                                 sizes[i] / size_t(a.cells / a.page_size), 0, sizes[i],
                    [&](size_t s, size_t d, size_t, size_t n) {
                        const auto* source = static_cast<const uint8_t*>(src[i]) + s;
                        auto* target = static_cast<uint8_t*>(dst[i]) + d;
                        return source == target || transfer(target, source, n, error);
                    })) return false;
        } else if (!transfer(dst[i], src[i], sizes[i], error)) return false;
    }
    return settle_resident(to, g, upto, error);
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !check_host_pages(st, l, error)) return false;
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    auto compare = [&](const void* device, const uint8_t* expected, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (!transfer(buffer.data(), static_cast<const uint8_t*>(device) + offset, n, error)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto authoritative = pools(st);
    for (size_t i = 0; i < saved.size(); ++i)
        if (!saved[i]->visit(0, saved[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                const auto* source = static_cast<const uint8_t*>(authoritative[i]);
                if (i < 4 && n && st.host.fp16_host_pages)
                    return visit_host_runs(st.host_page_ids, nullptr, saved[i]->size() / size_t(l.cells / l.page_size), at, n,
                        [&](size_t from, size_t, size_t logical, size_t run) {
                            return compare(source + from, p + logical - at, run, true);
                        });
                return compare(source + at, p, n, true);
            })) return false;
    if (st.kv_mode == 2 && image.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = image.cells / image.page_size;
        const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = saved[i]->size() / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(static_cast<const uint8_t*>(resident[i]) + offset, p, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

} // namespace basalt::core
