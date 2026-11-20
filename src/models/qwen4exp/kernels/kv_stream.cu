// src/models/qwen4exp/kernels/kv_stream.cu - see include/basalt/models/qwen4exp/kernels/kv_stream.hpp.
#include "basalt/models/qwen4exp/kernels/kv_stream.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_seg.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "kv_stream: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024;   // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
    const int32_t* host_pages;
};

// The INT8 pools' runs: codes, then the fp16 scales (host and device callers).
__host__ __device__ Runs runs_int8(const QsaAttnPools& slots, const KvHostPools& host, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
    Runs r{};
    r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
    r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
    r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
    r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
    r.n = 4;
    r.host_pages = host.fp16_host_pages;
    return r;
}

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        r = runs_int8(slots, host, s);
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
        r.host_pages = host.fp16_host_pages;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

// One map's resolve by the whole block (RT threads): resolve_kernel's body, and each block's of resolve_seg_kernel.
__device__ void resolve_block(const KvStreamMap& m, const int32_t* __restrict__ ids, const int32_t* __restrict__ steps,
                              int n_q, int cap, int page_size, QsaSegTable shared = {}, const int32_t* solo_pages = nullptr) {
    __shared__ int s_nmiss, s_lookups, s_cut;
    __shared__ int warp_sums[32];
    const int epoch = m.ctl[0] + 1;
    if (threadIdx.x == 0) { s_nmiss = 0; s_lookups = 0; }
    __syncthreads();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int32_t* pages = shared.shared_residency
            ? shared.slots[shared.rows[q].slot].host.fp16_host_pages : solo_pages;
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long) q * cap;
        for (int i = threadIdx.x; i < width; i += RT) {
            const int logical = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == logical) continue;   // ids are ascending: one lookup per block
            const int b = pages ? pages[logical] : logical;
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
            } else if (sl == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
                const int miss = atomicAdd(&s_nmiss, 1);
                if (miss >= m.n_slots) asm volatile("trap;");
                m.miss_block[miss] = b;
            }
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);   // most threads see none: skip the shared atomic
    __syncthreads();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        __syncthreads();
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    // Missing attention pages cannot be treated as zeros in the shared path.
    if ((shared.shared_residency || solo_pages) && placed < need) asm volatile("trap;");
    for (int k = threadIdx.x; k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

__global__ void __launch_bounds__(RT) resolve_kernel(KvStreamMap m, const int32_t* __restrict__ ids,
                                                     const int32_t* __restrict__ steps, int n_q, int cap, int page_size,
                                                     const int32_t* pages) {
    resolve_block(m, ids, steps, n_q, cap, page_size, {}, pages);
}

// Blocks blk, blk + nblk, ... of a copy: each copies one missed block's runs from the host copy into its slot, 16 B
// per thread.
__device__ void copy_misses(const KvStreamMap& m, const Runs& r, int blk, int nblk) {
    const int need = m.ctl[2];
    for (int k = blk; k < need; k += nblk) {
        const long long logical = m.miss_block[k], sl = m.miss_slot[k];
        const long long b = r.host_pages ? r.host_pages[logical] : logical;
        for (int a = 0; a < r.n; ++a) {
            const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + b * r.len[a]);
            uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + sl * r.len[a]);
            for (int i = threadIdx.x; i < r.len[a] / 16; i += blockDim.x) dst[i] = src[i];
        }
    }
}

__global__ void copy_kernel(KvStreamMap m, Runs r) { copy_misses(m, r, blockIdx.x, gridDim.x); }

// The indexer returns ascending cell IDs. Mark each page once per query, including a partially filled page.
__global__ void mark_prefill_pages(uint32_t* pages, const int32_t* ids, const int32_t* steps, int cap, int past) {
    const int q = blockIdx.x;
    const int width = steps[q * kStepCount + kStepWidth];
    ids += q * cap;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const int cell = ids[i];
        if (cell < 0 || cell >= past) continue;
        const unsigned page = (unsigned) cell >> 2;
        if (i == 0 || ((unsigned) ids[i - 1] >> 2) != page)
            atomicOr(pages + (page >> 5), 1u << (page & 31));
    }
}

__global__ void stage_prefill_pages(QsaAttnPools stage, QsaAttnPools resident, KvHostPools host,
                                    const uint32_t* pages, int n_pages) {
    constexpr int vectors = 2 * 4 * 256 * sizeof(uint16_t) / sizeof(uint4);
    auto* dst_k = reinterpret_cast<uint4*>(const_cast<uint16_t*>(stage.k_pool));
    auto* dst_v = reinterpret_cast<uint4*>(const_cast<uint16_t*>(stage.v_pool));
    for (int page = blockIdx.x; page < n_pages; page += gridDim.x) {
        if ((pages[page >> 5] & (1u << (page & 31))) == 0) continue;
        const int slot = resident.page_table[resident.physical_pages ? resident.physical_pages[page] : page];
        const auto* src_k = reinterpret_cast<const uint4*>(slot >= 0 ? resident.k_pool : host.k_pool);
        const auto* src_v = reinterpret_cast<const uint4*>(slot >= 0 ? resident.v_pool : host.v_pool);
        const int host_page = host.fp16_host_pages ? host.fp16_host_pages[page] : page;
        const int from = (slot >= 0 ? slot : host_page) * vectors, to = page * vectors;
        for (int i = threadIdx.x; i < vectors; i += blockDim.x) {
            dst_k[to + i] = src_k[from + i];
            dst_v[to + i] = src_v[from + i];
        }
    }
}

__global__ void stage_prefill_int8_pages(Runs runs, QsaAttnPools resident, const uint32_t* pages,
                                         int n_pages) {
    const void* gpu[] = {resident.k_q, resident.v_q, resident.k_scale, resident.v_scale};
    for (int page = blockIdx.x; page < n_pages; page += gridDim.x) {
        if ((pages[page >> 5] & (1u << (page & 31))) == 0) continue;
        const int slot = resident.page_table[resident.physical_pages ? resident.physical_pages[page] : page];
        const int host_page = runs.host_pages ? runs.host_pages[page] : page;
        for (int a = 0; a < runs.n; ++a) {
            const auto* source = slot >= 0 ? static_cast<const uint8_t*>(gpu[a]) : runs.src[a];
            const size_t offset = size_t(slot >= 0 ? slot : host_page) * runs.len[a];
            const auto* from = reinterpret_cast<const uint4*>(source + offset);
            auto* dst = reinterpret_cast<uint4*>(runs.dst[a] + size_t(page) * runs.len[a]);
            for (int j = threadIdx.x; j < runs.len[a] / int(sizeof(uint4)); j += blockDim.x)
                dst[j] = from[j];
        }
    }
}

// Block j: segment j's rows resolve into its slot's map.  A segment without a slot or rows, or a resident slot (null
// map), does nothing.  The segments name distinct slots, so no two blocks touch one map.
__global__ void __launch_bounds__(RT) resolve_seg_kernel(QsaSegTable t, const int32_t* __restrict__ ids,
                                                         const int32_t* __restrict__ steps, int cap, int page_size) {
    const QsaSeg sg = t.segs[blockIdx.x];
    if (sg.slot < 0 || sg.len <= 0) return;
    const KvStreamMap m = t.slots[sg.slot].map;
    if (m.ctl == nullptr) return;
    resolve_block(m, ids + (long long) sg.first * cap, steps + (long long) sg.first * kStepCount, sg.len, cap,
                  page_size);
}

// Grid (copy blocks, segments): segment blockIdx.y's misses into its slot's pool.
__global__ void __launch_bounds__(RT) resolve_shared_seg_kernel(
        QsaSegTable t, const int32_t* ids, const int32_t* steps, int cap, int page_size) {
    resolve_block(t.slots[0].map, ids, steps, t.n_rows, cap, page_size, t);
}

__global__ void copy_seg_kernel(QsaSegTable t, QsaShapes s) {
    const QsaSeg sg = t.segs[blockIdx.y];
    if (!t.shared_residency && (sg.slot < 0 || sg.len <= 0)) return;
    const QsaSegSlot& sl = t.slots[t.shared_residency ? 0 : sg.slot];
    if (sl.map.ctl == nullptr) return;
    Runs r{};
    if (sl.k_pool) {
        const int bytes = (int)(s.n_head_kv * s.page_size * s.head_dim * 2);
        r.src[0] = reinterpret_cast<const uint8_t*>(sl.host.k_pool);
        r.src[1] = reinterpret_cast<const uint8_t*>(sl.host.v_pool);
        r.dst[0] = reinterpret_cast<uint8_t*>(sl.k_pool);
        r.dst[1] = reinterpret_cast<uint8_t*>(sl.v_pool);
        r.len[0] = r.len[1] = bytes;
        r.n = 2;
        r.host_pages = sl.host.fp16_host_pages;
    } else {
        r = runs_int8(sl.pools(), sl.host, s);
    }
    if (t.shared_residency) r.host_pages = nullptr;  // misses are already physical
    copy_misses(sl.map, r, blockIdx.x, gridDim.x);
}

__global__ void reset_kernel(KvStreamMap m) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

__global__ void ring_kernel(int32_t* table, long long n_blocks, long long n_slots) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks;
         i += (long long) gridDim.x * blockDim.x)
        table[i] = (int32_t) (i % n_slots);
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

__global__ void invalidate_pages_kernel(KvStreamMap m, const int32_t* pages, int64_t count) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (int64_t) gridDim.x * blockDim.x) {
        const int page = pages[i];
        if (page < 0 || page >= m.n_blocks) asm volatile("trap;");
        const int slot = m.page_table[page];
        if (slot < 0) continue;
        m.page_table[page] = -1;
        m.slot_block[slot] = -1;
        m.slot_stamp[slot] = -1;
        m.slot_ref[slot] = 0;
    }
}

void kv_stream_invalidate(const KvStreamMap& m, const int32_t* physical_pages, int64_t count, void* stream) {
    if (count <= 0) return;
    invalidate_pages_kernel<<<64, 256, 0, (cudaStream_t) stream>>>(m, physical_pages, count);
    check("invalidate pages");
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    reset_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m);
    check("reset");
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        basalt::log::error("basalt", "kv_stream: a block's scale run must be a multiple of 16 bytes");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < kKvStreamMinSlots) {
        basalt::log::error("basalt", "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice", (long long) m.n_slots, RT);
        std::exit(1);
    }
    const auto* pages = slots.physical_pages;
    resolve_kernel<<<1, RT, 0, (cudaStream_t) stream>>>(m, ids, steps, (int) n_q, (int) cap, (int) s.page_size, pages);
    check("resolve");
    Runs runs = runs_of(slots, host, fmt, s);
    if (pages) runs.host_pages = nullptr;  // misses already hold physical IDs
    copy_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(m, runs);
    check("copy");
}

void kv_stream_resolve_seg(const QsaSegTable& t, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, void* stream) {
    static_assert(kKvStreamMinSlots == RT, "a streamed slot's pool holds at least one sweep step");
    if (t.n_seg <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        basalt::log::error("basalt", "kv_stream: a block's scale run must be a multiple of 16 bytes");
        std::exit(1);
    }
    // the slots' pool sizes live in the device table: the caller holds every streamed slot to kKvStreamMinSlots
    if (t.shared_residency)
        resolve_shared_seg_kernel<<<1, RT, 0, (cudaStream_t) stream>>>(t, ids, steps, (int) cap, (int) s.page_size);
    else
        resolve_seg_kernel<<<t.n_seg, RT, 0, (cudaStream_t) stream>>>(t, ids, steps, (int) cap, (int) s.page_size);
    check("resolve seg");
    copy_seg_kernel<<<dim3(96, t.shared_residency ? 1u : (unsigned) t.n_seg), 128, 0, (cudaStream_t) stream>>>(t, s);
    check("copy seg");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    ring_kernel<<<64, 256, 0, (cudaStream_t) stream>>>(page_table, n_blocks, n_slots);
    check("ring table");
}

// Mapped host backing is fragmented. Keep the existing bulk copies for contiguous host backing.
__global__ void copy_mapped_host(Runs r, int64_t b0, int64_t b1, int64_t n_slots, bool save) {
    for (int64_t logical = b0 + blockIdx.x; logical < b1; logical += gridDim.x) {
        const int64_t physical = r.host_pages[logical], slot = logical % n_slots;
        for (int a = 0; a < r.n; ++a) {
            auto* host = reinterpret_cast<uint4*>(const_cast<uint8_t*>(r.src[a]) + physical * r.len[a]);
            auto* device = reinterpret_cast<uint4*>(r.dst[a] + slot * r.len[a]);
            for (int i = threadIdx.x; i < r.len[a] / 16; i += blockDim.x) {
                if (save) host[i] = device[i];
                else device[i] = host[i];
            }
        }
    }
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    if (r.host_pages) {
        if (b1 > b0) copy_mapped_host<<<64, 128, 0, (cudaStream_t) stream>>>(r, b0, b1, n_slots, false);
        check("mapped ring restore");
        return;
    }
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring restore");
        b += run;
    }
}

void kv_ring_save(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1, int64_t n_slots,
                  const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);   // src: the host copy, dst: the ring
    if (r.host_pages) {
        if (b1 > b0) copy_mapped_host<<<64, 128, 0, (cudaStream_t) stream>>>(r, b0, b1, n_slots, true);
        check("mapped ring save");
        return;
    }
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync((void*) (r.src[a] + b * r.len[a]), r.dst[a] + sl * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring save");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    if (r.host_pages) {
        copy_mapped_host<<<64, 128, 0, (cudaStream_t) stream>>>(r, 0, n_blocks, n_blocks, false);
        check("mapped stage");
        return;
    }
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("stage");
}

bool kv_stage_selected(const QsaAttnPools& stage, const QsaAttnPools& resident, const KvHostPools& host,
                       int fmt, const QsaShapes& shapes, uint32_t* selected_pages,
                       const int32_t* ids, const int32_t* steps, int n_q, int cap, int past, void* stream) {
    if ((fmt != kKvF16 && fmt != kKvInt8) || shapes.n_head_kv != 2 ||
        shapes.page_size != 4 || shapes.head_dim != 256 || n_q <= 0 || cap <= 0 || past <= 0 ||
        !resident.page_table) return false;
    const Runs runs = runs_of(stage, host, fmt, shapes);
    for (int a = 0; a < runs.n; ++a)
        if (!runs.src[a] || !runs.dst[a]) return false;
    if (fmt == kKvInt8 ? !resident.k_q || !resident.v_q || !resident.k_scale || !resident.v_scale
                      : !resident.k_pool || !resident.v_pool) return false;
    const int n_pages = (past + 3) / 4;
    const cudaStream_t s = (cudaStream_t) stream;
    if (cudaMemsetAsync(selected_pages, 0, (size_t) ((n_pages + 31) / 32) * sizeof(uint32_t), s) != cudaSuccess)
        return false;
    mark_prefill_pages<<<n_q, 128, 0, s>>>(selected_pages, ids, steps, cap, past);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (fmt == kKvInt8)
        stage_prefill_int8_pages<<<512, 128, 0, s>>>(runs, resident, selected_pages, n_pages);
    else
        stage_prefill_pages<<<512, 128, 0, s>>>(stage, resident, host, selected_pages, n_pages);
    return cudaGetLastError() == cudaSuccess;
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace basalt::kernels
