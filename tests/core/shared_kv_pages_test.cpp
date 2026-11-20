#include "basalt/models/qwen4exp/decode/shared_kv_pages.hpp"
#include <cassert>
#include <set>

int main() {
    using Pool = basalt::core::SharedKvPages;
    Pool pool(131072);
    Pool::Lease long_request, next_request;
    assert(pool.reserve(long_request, 100000));
    assert(!pool.reserve(next_request, 31073));
    assert(pool.available() == 31072);
    assert(!pool.materialize(long_request, 100001));
    assert(pool.reserve(next_request, 31072));
    assert(pool.materialize(next_request, 31072));
    assert(pool.materialize(long_request, 100000));
    std::set<uint32_t> owned(long_request.pages().begin(), long_request.pages().end());
    for (auto page : next_request.pages()) assert(owned.insert(page).second);
    assert(owned.size() == pool.capacity());
    pool.release(long_request);
    pool.release(long_request);
    assert(pool.available() == 100000);
    assert(pool.reserve(long_request, 100000));
    assert(pool.materialize(long_request, 100000));
    for (auto page : next_request.pages()) assert(!std::ranges::binary_search(long_request.pages(), page));
    pool.release(long_request);
    pool.release(next_request);
    assert(pool.available() == pool.capacity());
    assert(pool.reserve(long_request, pool.capacity()));
    pool.release(long_request);
    assert(pool.available() == pool.capacity());

    assert(pool.reserve(long_request, 100000));
    assert(pool.materialize(long_request, 64));
    const auto* address = long_request.pages().data();
    const auto first = long_request.pages().front();
    assert(!pool.resize_reservation(long_request, 63));
    assert(pool.resize_reservation(long_request, 64));
    assert(pool.available() == pool.capacity() - 64);
    assert(long_request.pages().data() == address);
    assert(pool.reserve(next_request, pool.available()));
    assert(!pool.resize_reservation(long_request, 65));
    assert(pool.available() == 0 && long_request.pages().front() == first);
    pool.release(next_request);
    assert(pool.resize_reservation(long_request, 100000));
    assert(pool.materialize(long_request, 65));
    assert(long_request.pages().front() == first);
    const auto discarded = long_request.pages().back();
    assert(!pool.trim(long_request, 66));
    assert(pool.trim(long_request, 64));
    assert(pool.available() == pool.capacity() - 64);
    assert(pool.reserve(next_request, 1) && pool.materialize(next_request, 1));
    assert(next_request.pages().front() == discarded);
    pool.release(next_request);
    pool.release(long_request);
    assert(pool.available() == pool.capacity());
}
