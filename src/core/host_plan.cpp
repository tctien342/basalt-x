// src/core/host_plan.cpp - see include/basalt/core/host_plan.hpp.
#include "basalt/core/host_plan.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/memory.hpp"

#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace basalt::core {

namespace {
std::mutex g_mu;
std::vector<std::pair<void*, size_t>> g_held;
}  // namespace

bool host_plan(size_t bytes, HostPlanTouch touch) {
    if (bytes == 0) return true;
    void* p = platform::alloc_aligned(bytes, 4096);
    if (p == nullptr) return false;
    const size_t page = touch == HostPlanTouch::Zero ? 0 : platform::system_page_size();
    bool prefaulted = false;
#if defined(__linux__) && defined(MADV_POPULATE_WRITE)
    if (touch == HostPlanTouch::Prefault && page != 0 && reinterpret_cast<uintptr_t>(p) % page == 0)
        prefaulted = madvise(p, bytes, MADV_POPULATE_WRITE) == 0;
#endif
    if (prefaulted) {
        basalt::log::info("basalt startup", "host reserve prefaulted %llu B", (unsigned long long) bytes);
    } else if (page != 0) {
        auto* data = static_cast<volatile uint8_t*>(p);
        for (size_t at = 0; at < bytes; at += page) data[at] = 0;
        data[bytes - 1] = 0;  // Include the last page if the returned address is not page-aligned.
    } else {
        std::memset(p, 0, bytes);
    }
    const std::lock_guard<std::mutex> lk(g_mu);
    g_held.emplace_back(p, bytes);
    return true;
}

cudaError_t host_alloc(void** p, size_t bytes, unsigned flags) {
    {
        const std::lock_guard<std::mutex> lk(g_mu);
        size_t pick = g_held.size();
        for (size_t i = 0; i < g_held.size(); ++i) {
            const size_t b = g_held[i].second;
            if (pick == g_held.size()) { pick = i; continue; }
            const size_t c = g_held[pick].second;
            if ((b >= bytes && (c < bytes || b < c)) || (c < bytes && b > c)) pick = i;
        }
        if (pick < g_held.size()) {
            platform::free_aligned(g_held[pick].first);
            g_held.erase(g_held.begin() + (std::ptrdiff_t) pick);
        }
    }
    return cudaHostAlloc(p, bytes, flags);
}

uint64_t host_plan_bytes() {
    const std::lock_guard<std::mutex> lk(g_mu);
    uint64_t n = 0;
    for (const auto& h : g_held) n += h.second;
    return n;
}

uint64_t host_plan_release() {
    const std::lock_guard<std::mutex> lk(g_mu);
    uint64_t n = 0;
    for (const auto& h : g_held) {
        platform::free_aligned(h.first);
        n += h.second;
    }
    g_held.clear();
    return n;
}

}  // namespace basalt::core
