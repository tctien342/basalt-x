#pragma once

#include <cstdint>
#include <list>
#include <stdexcept>

#include "basalt/models/qwen4exp/decode/shared_kv_pages.hpp"

namespace basalt::core {

// Only the scheduler accesses pending tickets. Running leases stay with their requests.
class SharedKvAdmission {
public:
    using Ticket = uint64_t;

    explicit SharedKvAdmission(SharedKvPages& pool) : pool_(pool) {}

    SharedKvAdmission(const SharedKvAdmission&) = delete;
    SharedKvAdmission& operator=(const SharedKvAdmission&) = delete;

    Ticket enqueue(uint32_t pages, SharedKvPages::Lease& lease) {
        if (!pages || pages > pool_.capacity()) throw std::invalid_argument("request exceeds KV pool capacity");
        const Ticket id = ++next_;
        pending_.push_back({id, pages, &lease});
        return id;
    }

    bool cancel(Ticket id) {
        for (auto it = pending_.begin(); it != pending_.end(); ++it) {
            if (it->id != id) continue;
            pending_.erase(it);
            return true;
        }
        return false;
    }

    // The caller supplies the final token count after media expansion. Use
    // subtraction checks so oversized caps cannot wrap before admission.
    Ticket enqueue_tokens(int64_t prompt, int64_t output, uint32_t page_cells, SharedKvPages::Lease& lease) {
        const uint64_t capacity = uint64_t(pool_.capacity()) * page_cells;
        if (capacity > INT64_MAX) throw std::invalid_argument("invalid KV pool geometry");
        const auto limit = static_cast<int64_t>(capacity);
        constexpr int64_t slack = 8;  // native request context guard
        if (!page_cells || prompt < 0 || output < 1 || limit < slack ||
            prompt > limit - slack || output > limit - slack - prompt)
            throw std::invalid_argument("request exceeds KV pool capacity");
        const int64_t cells = prompt + output + slack;
        return enqueue(uint32_t(1 + (cells - 1) / page_cells), lease);
    }

    // Return zero when no request fits. Protect the oldest blocked request after one bypass.
    Ticket admit_next() {
        for (auto it = pending_.begin(); it != pending_.end(); ++it) {
            if (pool_.reserve(*it->lease, it->pages)) {
                const Ticket id = it->id;
                if (it != pending_.begin()) pending_.front().bypassed = true;
                pending_.erase(it);
                return id;
            }
            if (it->bypassed) break;
        }
        return 0;
    }

private:
    struct Pending {
        Ticket id;
        uint32_t pages;
        SharedKvPages::Lease* lease;
        bool bypassed = false;
    };
    SharedKvPages& pool_;
    Ticket next_ = 0;
    std::list<Pending> pending_;
};

}  // namespace basalt::core
