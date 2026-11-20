#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

int MtpDrafter::kv_layout(KvArr* out) const {
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const uint64_t rows = (uint64_t) s.n_head_kv * (uint64_t) s.page_size;
    const uint64_t hd = (uint64_t) s.head_dim;
    if (st_.kv_hybrid) return 0;
    if (st_.kv_q4) {
        const uint64_t q4 = rows * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim);
        out[0] = {st_.k_q4, q4};
        out[1] = {st_.v_q4, q4};
        return 2;
    }
    if (st_.kv_int8) {
        const uint64_t sc = rows * (hd / basalt::kernels::KV_Q8_GROUP) * 2;
        out[0] = {st_.k_q, rows * hd};
        out[1] = {st_.v_q, rows * hd};
        out[2] = {st_.k_scale, sc};
        out[3] = {st_.v_scale, sc};
        return 4;
    }
    out[0] = {st_.k_pool, rows * hd * 2};
    out[1] = {st_.v_pool, rows * hd * 2};
    return 2;
}

bool MtpDrafter::ensure_store(int id, const KvArr* layout, int arrays, std::string& err, cudaError_t* status) {
    auto& store = stores_[(size_t) id];
    if (store[0] || arrays == 0) return true;
    std::array<void*, 4> allocated{};
    uint64_t bytes_total = 0;
    for (int i = 0; i < arrays; ++i) {
        const auto bytes = (uint64_t) store_ring_[(size_t) id] * layout[i].page_bytes;
        auto& ptr = allocated[(size_t) i];
        const auto alloc = cudaMalloc(&ptr, bytes);
        const auto init = alloc == cudaSuccess ? cudaMemset(ptr, 0, bytes) : cudaSuccess;
        if (alloc != cudaSuccess || init != cudaSuccess) {
            if (status) *status = alloc == cudaSuccess ? init : alloc;
            size_t free = 0, total = 0;
            const bool have_memory = cudaMemGetInfo(&free, &total) == cudaSuccess;
            for (auto* buffer : allocated) if (buffer) cudaFree(buffer);
            err = std::string("mtp: allocating the slot's draft K/V failed: ") +
                  (alloc == cudaSuccess ? "cudaMemset: " : "cudaMalloc: ") +
                  cudaGetErrorString(alloc == cudaSuccess ? init : alloc) +
                  " (array " + std::to_string(i) + ", " + std::to_string(bytes) + " bytes";
            if (have_memory) err += ", " + std::to_string(free) + " free";
            err += ")";
            return false;
        }
        bytes_total += bytes;
    }
    store = allocated;
    vram_ += bytes_total;
    return true;
}

void MtpDrafter::note_cells(int64_t lo, int64_t hi) {
    hi = std::min(hi, st_.max_cells);
    if (hi <= lo) return;
    if (dirty_hi_ <= dirty_lo_) { dirty_lo_ = lo; dirty_hi_ = hi; }
    else { dirty_lo_ = std::min(dirty_lo_, lo); dirty_hi_ = std::max(dirty_hi_, hi); }
    if (store_hi_.empty()) return;
    // the sequence ends where its latest write ends; a write that starts well below the end (another prompt, a
    // rolled-back turn) starts it again there, past the cells a window writes ahead of its position
    int64_t& end = store_hi_[(size_t) cur_];
    end = lo + 4 * (int64_t) max_t_ + 64 < end ? hi : std::max(end, hi);
}

void MtpDrafter::copy_pages(int to, bool to_work, int from, bool from_work, int64_t a, int64_t b, const KvArr* lay,
                            int na) {
    // the working state holds every page in place, or with KV streaming its ring (`block % n_slots`), which reaches as
    // far back as any store (load)
    const int64_t work_ring = st_.kv_mode == 2 ? st_.n_slots : 0;
    auto at = [&](int id, bool work, int i, int64_t p) {
        const int64_t ring = work ? work_ring : store_ring_[(size_t) id];
        return (uint8_t*) (work ? lay[i].p : stores_[(size_t) id][(size_t) i]) +
               (uint64_t) (ring > 0 ? p % ring : p) * lay[i].page_bytes;
    };
    auto run = [&](int id, bool work, int64_t p) {   // pages from p before that side's ring wraps
        const int64_t ring = work ? work_ring : store_ring_[(size_t) id];
        return ring > 0 ? ring - p % ring : b - p;
    };
    for (int64_t p = a; p < b;) {
        const int64_t n = std::min({b - p, run(to, to_work, p), run(from, from_work, p)});
        for (int i = 0; i < na; ++i)
            cudaMemcpyAsync(at(to, to_work, i, p), at(from, from_work, i, p), (size_t) ((uint64_t) n * lay[i].page_bytes),
                            cudaMemcpyDeviceToDevice, cs_);
        p += n;
    }
}

uint64_t MtpDrafter::shared_store_bytes(int slot) const {
    KvArr layout[4];
    uint64_t page_bytes = 0;
    const int arrays = kv_layout(layout);
    for (int i = 0; i < arrays; ++i) page_bytes += layout[i].page_bytes;
    if (slot < 0) return (uint64_t) store_ring_.back() * page_bytes;
    const auto b = (size_t) slot;
    const auto main = (size_t) n_stores_;
    return page_bytes * ((stores_[main][0] ? 0 : store_ring_[main]) +
                         (stores_[b][0] ? 0 : store_ring_[b]));
}

bool MtpDrafter::prepare_shared_stores(int slot, std::string& err) {
    const OnDevice on_device(device_);
    KvArr layout[4];
    const int arrays = kv_layout(layout);
    const bool ok = ensure_store(n_stores_, layout, arrays, err) &&
                    ensure_store(slot, layout, arrays, err);
    if (!ok) (void) cudaGetLastError();
    return ok;
}

bool MtpDrafter::append_shared_store(int64_t cells, std::string& err, cudaError_t& status) {
    status = cudaSuccess;
    if (!lazy_stores_ || n_stores_ <= 0 || pend_R_.empty() || cells <= 0 || cells > st_.max_cells) {
        err = "mtp: invalid shared store geometry";
        return false;
    }
    const OnDevice on_device(device_);
    const auto count = stores_.size() + 1;
    stores_.reserve(count);
    store_hi_.reserve(count);
    store_pages_.reserve(count);
    store_ring_.reserve(count);
    store_adapter_scale_.reserve(count);
    pend_R_.reserve(count);
    pend_tok_.reserve(count);
    pend_cell0_.reserve(count);
    const size_t main = (size_t) n_stores_;
    const int64_t ps = shapes_of(*g_).page_size;
    const int64_t pages = 1 + (cells - 1) / ps;
    // Prove the new slot can store its ring before reporting successful growth.
    KvArr layout[4];
    const int arrays = kv_layout(layout);
    const auto old_ring = store_ring_[main];
    const auto old_store = stores_[main];
    stores_[main] = {};
    store_ring_[main] = std::min(old_ring, pages);
    if (!ensure_store((int) main, layout, arrays, err, &status)) {
        stores_[main] = old_store;
        store_ring_[main] = old_ring;
        cudaGetLastError();
        return false;
    }
    const auto slot_store = stores_[main];
    // Keep existing slot IDs. The main store moves to the new last index.
    stores_.push_back(old_store);
    stores_[main] = slot_store;
    store_hi_.push_back(store_hi_[main]);
    store_hi_[main] = 0;
    store_pages_.push_back(store_pages_[main]);
    store_pages_[main] = pages;
    store_ring_.push_back(old_ring);
    store_adapter_scale_.push_back(store_adapter_scale_[main]);
    store_adapter_scale_[main] = 0;
    pend_R_.push_back(pend_R_[main]);
    pend_R_[main] = nullptr;
    pend_tok_.emplace_back();
    pend_tok_.back().swap(pend_tok_[main]);
    pend_cell0_.push_back(pend_cell0_[main]);
    pend_cell0_[main] = 0;
    if (cur_ == n_stores_) ++cur_;
    ++n_stores_;
    return true;
}

bool MtpDrafter::retire_shared_store(int id, std::string& err) {
    if (!lazy_stores_ || id < 0 || id >= n_stores_) return true;
    const OnDevice on_device(device_);
    if (!idle(err)) return false;
    const size_t slot = (size_t) id;
    store_hi_[slot] = 0;
    pend_tok_[slot].clear();
    if (cur_ == id) dirty_lo_ = dirty_hi_ = 0;

    // Keep the admitted ring available for the next request on this slot.
    if (auto& rows = pend_R_[slot]; rows) {
        if (const auto status = cudaFree(rows); status != cudaSuccess) {
            err = cudaGetErrorString(status);
            return false;
        }
        rows = nullptr;
        vram_ -= (uint64_t) kPendRows * g_->hc * g_->n_embd * sizeof(float);
    }
    return true;
}

bool MtpDrafter::set_store_adapter_scale(int id, float scale, std::string& err) {
    if (!h_adapter_rows_ || store_adapter_scale_[(size_t) id] == scale) return true;
    if (id == cur_) {
        const OnDevice on_device(device_);
        if (const auto status = cudaStreamSynchronize(cs_); status != cudaSuccess) {
            err = cudaGetErrorString(status);
            return false;
        }
        std::fill_n(h_adapter_rows_, max_t_, scale);
    }
    store_adapter_scale_[(size_t) id] = scale;
    return true;
}

bool MtpDrafter::use_store(int id, std::string& err) {
    if (id == cur_) return true;
    if (id < 0 || id > n_stores_) { err = "mtp: no such K/V store"; return false; }
    const OnDevice on_device(device_);
    KvArr lay[4];
    const int na = kv_layout(lay);
    const int64_t ps = shapes_of(*g_).page_size;
    if (dirty_hi_ > dirty_lo_) {   // the written pages go back to the sequence that held them, as far as its ring reaches
        const size_t c = (size_t) cur_;
        const int64_t pg1 = std::min<int64_t>(store_pages_[c], (std::min(dirty_hi_, store_hi_[c]) + ps - 1) / ps);
        if (!ensure_store(cur_, lay, na, err)) return false;
        copy_pages(cur_, false, cur_, true, std::max<int64_t>(dirty_lo_ / ps, pg1 - store_ring_[c]), pg1, lay, na);
    }
    // the new sequence's last pages: all its window reads
    const int64_t n1 = std::min<int64_t>(store_pages_[(size_t) id], (store_hi_[(size_t) id] + ps - 1) / ps);
    if (n1 > 0 && !ensure_store(id, lay, na, err)) return false;
    copy_pages(id, true, id, false, std::max<int64_t>(0, n1 - store_ring_[(size_t) id]), n1, lay, na);
    if (h_adapter_rows_ && store_adapter_scale_[(size_t) cur_] != store_adapter_scale_[(size_t) id]) {
        if (const auto status = cudaStreamSynchronize(cs_); status != cudaSuccess) {
            err = cudaGetErrorString(status);
            return false;
        }
        std::fill_n(h_adapter_rows_, max_t_, store_adapter_scale_[(size_t) id]);
    }
    cur_ = id;
    dirty_lo_ = dirty_hi_ = 0;
    if (cudaGetLastError() != cudaSuccess) { err = "mtp: swapping the K/V failed"; return false; }
    return true;
}

bool MtpDrafter::copy_store(int from, int to, int64_t cells, std::string& err) {
    if (n_stores_ == 0 || from == to) return true;
    if (from < 0 || from > n_stores_ || to < 0 || to > n_stores_) { err = "mtp: no such K/V store"; return false; }
    if (!flush(from, err)) return false;
    pend_tok_[(size_t) to].clear();
    const OnDevice on_device(device_);
    KvArr lay[4];
    const int na = kv_layout(lay);
    const int64_t ps = shapes_of(*g_).page_size;
    const int64_t n1 = std::min<int64_t>({store_pages_[(size_t) from], store_pages_[(size_t) to], (cells + ps - 1) / ps});
    // the last pages both sides keep (a store its ring, the working state all)
    const int64_t keep = std::min(from == cur_ ? n1 : store_ring_[(size_t) from], to == cur_ ? n1 : store_ring_[(size_t) to]);
    if (n1 > 0 && ((from != cur_ && !ensure_store(from, lay, na, err)) ||
                  (to != cur_ && !ensure_store(to, lay, na, err)))) return false;
    copy_pages(to, to == cur_, from, from == cur_, std::max<int64_t>(0, n1 - keep), n1, lay, na);
    if (to == cur_) note_cells(0, cells);
    store_hi_[(size_t) to] = std::min<int64_t>({cells, st_.max_cells, store_pages_[(size_t) to] * ps});
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: copying the K/V failed"; return false; }
    return true;
}

bool MtpDrafter::save_store(int id, int64_t cells, ConversationKv& out, std::string& err) {
    if (n_stores_ == 0) return conversation_kv_save(out, st_, *g_, cells, false, err);
    if (id < 0 || id > n_stores_) { err = "mtp: no such K/V store"; return false; }
    const int prev = cur_;
    if (!pend_tok_[(size_t) id].empty()) {
        // its deferred rows are computed in the working state; the sequence there comes back whole within its ring
        note_cells(0, store_hi_[(size_t) prev]);
        if (!flush(id, err)) return false;
    }
    // the image comes from the host copy, which a ring shares with every sequence: the one it holds puts its pages
    // back first (another sequence's come from its store, below)
    if (cur_ == id && !ring_to_host(err)) return false;
    if (!conversation_kv_save(out, st_, *g_, cells, false, err)) return false;
    if (cur_ != id) {   // the pages the store keeps, over the working state's
        const OnDevice on_device(device_);
        KvArr lay[4];
        const int na = kv_layout(lay);
        ConversationBuffer* dst[4] = {&out.k, &out.v, &out.k_scale, &out.v_scale};
        const int64_t ps = shapes_of(*g_).page_size, ring = store_ring_[(size_t) id];
        const int64_t n1 = std::min<int64_t>(store_pages_[(size_t) id], (cells + ps - 1) / ps);
        if (n1 > 0 && !ensure_store(id, lay, na, err)) return false;
        for (int64_t p = std::max<int64_t>(0, n1 - ring); p < n1;) {
            const int64_t n = std::min(n1 - p, ring - p % ring);
            for (int i = 0; i < na; ++i) {
                const uint8_t* src = (const uint8_t*) stores_[(size_t) id][(size_t) i] +
                                     (uint64_t) (p % ring) * lay[i].page_bytes;
                const size_t off = (size_t) ((uint64_t) p * lay[i].page_bytes);
                if (!dst[i]->visit(off, (size_t) ((uint64_t) n * lay[i].page_bytes), [&](uint8_t* q, size_t m, size_t at) {
                        return cudaMemcpy(q, src + (at - off), m, cudaMemcpyDeviceToHost) == cudaSuccess;
                    })) {
                    err = "mtp: saving the K/V failed";
                    return false;
                }
            }
            p += n;
        }
    }
    return cur_ == prev || use_store(prev, err);
}

bool MtpDrafter::before_draft_write(int64_t first, int64_t end, std::string& err) {
    if (!frozen_draft_ready_.valid() || first >= end || end <= frozen_draft_first_ ||
        first >= frozen_draft_end_) return true;
    try {
        if (frozen_draft_ready_.get()) {
            frozen_draft_ready_ = {};
            return true;
        }
    } catch (const std::future_error&) {
        // The recovery worker exited before making its copy ready.
    }
    err = "mtp: frozen draft history copy failed";
    return false;
}

bool MtpDrafter::ring_to_host(std::string& err) {
    if (st_.kv_mode != 2 || n_stores_ == 0) return true;
    const OnDevice on_device(device_);
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const size_t c = (size_t) cur_;
    const int64_t n1 = std::min<int64_t>(store_pages_[c], (store_hi_[c] + s.page_size - 1) / s.page_size);
    const int64_t first = std::max<int64_t>(0, n1 - store_ring_[c]);
    if (!before_draft_write(first * s.page_size, n1 * s.page_size, err)) return false;
    basalt::kernels::kv_ring_save(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), first, n1, st_.n_slots, s, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: saving the draft ring failed"; return false; }
    return true;
}

QsaState& MtpDrafter::kv_state_rw() {
    std::string e;
    if (!use_main(e)) basalt::log::error("basalt mtp", "%s", e.c_str());
    return st_;
}


}
