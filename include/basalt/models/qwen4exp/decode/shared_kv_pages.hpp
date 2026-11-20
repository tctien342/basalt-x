#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace basalt::core {

// Single scheduler owner. Release only after all GPU readers and writers drain.
class SharedKvPages {
public:
    explicit SharedKvPages(uint32_t pages) : capacity_(pages), available_(pages) {
        free_.reserve(pages);
        for (uint32_t i = pages; i > 0; --i) free_.push_back(i - 1);
    }

    SharedKvPages(const SharedKvPages&) = delete;
    SharedKvPages& operator=(const SharedKvPages&) = delete;

    struct Lease {
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        std::span<const uint32_t> pages() const { return pages_; }
    private:
        friend class SharedKvPages;
        SharedKvPages* owner_ = nullptr;
        uint32_t reserved_ = 0;
        std::vector<uint32_t> pages_;
    };

    uint32_t available() const { return available_; }
    uint32_t capacity() const { return capacity_; }

    bool reserve(Lease& lease, uint32_t count) {
        if (lease.owner_ || !count || count > available_) return false;
        lease.pages_.reserve(count);
        lease.owner_ = this;
        lease.reserved_ = count;
        available_ -= count;
        return true;
    }

    // Return unused reservations while keeping every materialized prefix page.
    // A larger continuation may reclaim capacity without moving its pages.
    bool resize_reservation(Lease& lease, uint32_t count) {
        if (lease.owner_ != this || !count || count < lease.pages_.size()) return false;
        if (count > lease.reserved_ && count - lease.reserved_ > available_) return false;
        lease.pages_.reserve(count);
        available_ = available_ + lease.reserved_ - count;
        lease.reserved_ = count;
        return true;
    }

    bool materialize(Lease& lease, uint32_t target) {
        if (lease.owner_ != this || target > lease.reserved_) return false;
        while (lease.pages_.size() < target) {
            assert(!free_.empty());
            lease.pages_.push_back(free_.back());
            free_.pop_back();
        }
        return true;
    }

    // The caller invalidates the discarded pages on all devices before trimming.
    bool trim(Lease& lease, uint32_t keep) {
        if (lease.owner_ != this || !keep || keep > lease.pages_.size()) return false;
        free_.insert(free_.end(), lease.pages_.rbegin(), lease.pages_.rend() - keep);
        lease.pages_.resize(keep);
        available_ += lease.reserved_ - keep;
        lease.reserved_ = keep;
        return true;
    }

    void release(Lease& lease) {
        if (lease.owner_ != this) return;
        free_.insert(free_.end(), lease.pages_.rbegin(), lease.pages_.rend());
        available_ += lease.reserved_;
        lease.pages_.clear();
        lease.reserved_ = 0;
        lease.owner_ = nullptr;
    }

private:
    const uint32_t capacity_;
    uint32_t available_;
    std::vector<uint32_t> free_;
};

}  // namespace basalt::core
