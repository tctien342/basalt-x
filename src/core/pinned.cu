// src/core/pinned.cu - the pinned host arena.
#include "basalt/core/pinned.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
#include "basalt/log.hpp"

namespace basalt::core {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         const std::string& shared_file, uint64_t shared_hash,
                         const std::function<bool()>& before_register) : capacity(bytes) {
    if (bytes == 0) return;
    basalt::platform::RegionBacking region_backing = basalt::platform::RegionBacking::NormalPages;
    base = basalt::platform::alloc_region(bytes, region_backing, note, shared_file, shared_hash, mapping_base,
                                          mapping_bytes, static_cast<bool>(before_register));
    backing = region_backing == basalt::platform::RegionBacking::LargePages ? PageBacking::LargePages
                                                                             : PageBacking::NormalPages;

    bool permit_checked = false;
    bool permitted = false;
#if defined(__linux__) && defined(MADV_POPULATE_READ)
    if (base && !shared_file.empty() && exp_flag("startup_shared_arena_read_prefault") != 0.0 &&
        basalt::platform::shared_arena_ready(mapping_base)) {
        const auto start = std::chrono::steady_clock::now();
        // Break up the worker's hold on the mapping lock so pinned allocations can proceed.
        const uint64_t piece = before_register ? 256ull << 20 : bytes;
        const uint64_t page = before_register ? basalt::platform::system_page_size() : 0;
        std::vector<unsigned char> residency(page ? (piece + page - 1) / page : 0);
        const auto advise_span = [&](uint64_t begin, uint64_t end, bool readahead = false) -> std::pair<uint64_t, int> {
            for (uint64_t off = begin; off < end;) {
                const uint64_t count = std::min(piece, end - off);
                auto* at = static_cast<uint8_t*>(base) + off;
                if (readahead) (void) madvise(at, (size_t) count, MADV_WILLNEED);
                if (madvise(at, (size_t) count, MADV_POPULATE_READ) != 0)
                    return {off - begin, errno};
                off += count;
            }
            return {end - begin, 0};
        };
        int error = 0;
        uint64_t done = 0;
        while (done < bytes) {
            const uint64_t count = std::min(piece, bytes - done);
            auto* at = static_cast<uint8_t*>(base) + done;
            if (before_register && !permit_checked) {
                const size_t pages = page ? (count + page - 1) / page : 0;
                const bool resident = pages && mincore(at, (size_t) count, residency.data()) == 0 &&
                    std::all_of(residency.begin(), residency.begin() + pages, [](unsigned char v) { return v & 1; });
                if (!resident) {
                    // Leave page-in I/O until the main startup thread has loaded the draft and head.
                    basalt::log::info("basalt startup", "shared arena early prefault deferred at %llu bytes",
                                      (unsigned long long) done);
                    permit_checked = true;
                    permitted = before_register();
                    if (!permitted) { error = ECANCELED; break; }
                    const uint64_t half_pieces = ((bytes - done + piece - 1) / piece) / 2;
                    if (half_pieces > 0 && exp_flag("startup_arena_cold_two_prefault") != 0.0) {
                        const uint64_t split = done + half_pieces * piece;
                        const bool readahead = exp_flag("startup_arena_cold_readahead") != 0.0;
                        std::pair<uint64_t, int> low{}, high{};
                        bool parallel = false;
                        try {
                            std::jthread reader([&] { high = advise_span(split, bytes, readahead); });
                            low = advise_span(done, split, readahead);
                            parallel = true;
                        } catch (const std::system_error&) {
                            low = advise_span(done, bytes, readahead);
                        }
                        done += low.first + high.first;
                        error = low.second != 0 ? low.second : high.second;
                        basalt::log::info("basalt startup", "shared arena deferred prefault: %d lanes, readahead=%d",
                                          parallel ? 2 : 1, int(readahead));
                        break;
                    }
                }
            }
            const auto [advised, status] = advise_span(done, done + count);
            done += advised;
            error = status;
            if (error) break;
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        basalt::log::info("basalt startup", "shared arena read-prefault: %.3f s, bytes %llu, error %d",
                         elapsed, (unsigned long long) bytes, error);
        if (before_register)
            basalt::log::info("basalt startup", "shared arena early prefault: 256 MiB batches, %llu bytes advised",
                              (unsigned long long) done);
    }
#endif
    if (base && before_register && !(permit_checked ? permitted : before_register())) {
        basalt::platform::free_region(mapping_base, mapping_bytes);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
        note = "shared arena registration canceled";
        return;
    }
    // Cold arenas register before population. Faulting their payload from user space first can make registration
    // much more expensive or cause it to fail.
    if (base) {
        using Clock = std::chrono::steady_clock;
        const auto register_start = before_register ? Clock::now() : Clock::time_point{};
        const RamPin p = pin_ram(base, bytes, bytes, bounds, /*mapped=*/true);
        if (before_register)
            basalt::log::info("basalt startup", "shared arena early CUDA registration: %.3f s",
                              std::chrono::duration<double>(Clock::now() - register_start).count());
        registered_bytes = p.registered_bytes;
        locked_bytes = p.locked_bytes;
        if (p.refused.empty()) {
            note = "cudaHostRegister PORTABLE ok; " + note;
        } else if (p.sliced) {
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            slice_starts = p.slice_starts;
            registered_slices = p.slices;
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) p.cap / (double) (1ull << 30));
            note = "cudaHostRegister of the whole arena FAILED (" + p.refused + "); " +
                   (p.capped ? "slices capped at " + std::string(gib) + " GiB (" + p.cap_why + "); " : std::string()) +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB); " + note;
            if (registered_bytes < bytes) note = p.lock_note + "; " + note;
        } else {
            note = "cudaHostRegister FAILED (" + p.refused + ") - the arena is NOT pinned, so copies will be slow; " + note;
            note = p.lock_note + "; " + note;
        }
    }
}

PinnedArena::~PinnedArena() {
    if (base) {
        if (locked_bytes) basalt::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else if (!platform::kBoundHostRegistration || registered_bytes > 0) {
            cudaHostUnregister(base);
        }
        basalt::platform::free_region(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        basalt::log::error("basalt", "stream_bandwidth: cudaMalloc failed for %llu B", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace basalt::core
