#include "basalt/models/qwen4exp/decode/shared_kv_admission.hpp"

#include <cassert>

int main() {
    using namespace basalt::core;
    SharedKvPages pool(131072);
    SharedKvAdmission queue(pool);
    SharedKvPages::Lease active, large, small, later, canceled;
    assert(pool.reserve(active, 120000));
    auto head = queue.enqueue(100000, large);
    assert(queue.admit_next() == 0);
    auto skip = queue.enqueue(4096, small);
    assert(queue.admit_next() == skip);
    auto tail = queue.enqueue(4096, later);
    auto drop = queue.enqueue(1024, canceled);
    assert(queue.cancel(drop));
    assert(!queue.cancel(drop));
    assert(queue.admit_next() == 0);
    assert(pool.available() == 6976);
    pool.release(active);
    assert(queue.admit_next() == head);
    assert(queue.admit_next() == tail);
    assert(queue.admit_next() == 0);
    try {
        queue.enqueue(131073, canceled);
        assert(false);
    } catch (const std::invalid_argument&) {}
    pool.release(large);
    pool.release(small);
    pool.release(later);
    assert(pool.available() == pool.capacity());

    const auto first = queue.enqueue_tokens(399988, 4, 4, active);
    assert(queue.admit_next() == first && pool.available() == 31072);
    const auto exact = queue.enqueue_tokens(124277, 3, 4, small);
    assert(queue.admit_next() == exact && pool.available() == 0);
    pool.release(small);
    const auto fits = queue.enqueue_tokens(124278, 3, 4, large);
    assert(queue.admit_next() == 0);
    pool.release(active);
    assert(queue.admit_next() == fits);
    assert(pool.available() == 99999);
    pool.release(large);
    try {
        queue.enqueue_tokens(1, INT64_MAX, 4, canceled);
        assert(false);
    } catch (const std::invalid_argument&) {}
}
