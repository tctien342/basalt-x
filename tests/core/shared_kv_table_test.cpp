#include "basalt/models/qwen4exp/decode/shared_kv_table.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace {
int current_device = 0;
bool fail_peer_sync = true, fail_peer_copy = false;
int invalidations[2] = {};
}

extern "C" cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind) {
    if (current_device == 1 && fail_peer_copy) return cudaErrorUnknown;
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
extern "C" cudaError_t cudaGetDevice(int* device) {
    *device = current_device;
    return cudaSuccess;
}
extern "C" cudaError_t cudaSetDevice(int device) {
    current_device = device;
    return cudaSuccess;
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return current_device == 1 && fail_peer_sync ? cudaErrorUnknown : cudaSuccess;
}
namespace basalt::kernels {
void kv_stream_invalidate(const KvStreamMap&, const int32_t* pages, int64_t count, void*) {
    assert((count == 3 && pages[0] == 0 && pages[2] == 2) || (count == 1 && pages[0] == 2) || (count == 2 && pages[0] == 0 && pages[1] == 1));
    ++invalidations[current_device];
}
}

int main() {
    using namespace basalt::core;
    SharedKvPages pool(8);
    SharedKvPages::Lease lease, other;
    assert(pool.reserve(lease, 6) && pool.materialize(lease, 2));
    SharedKvTable tables[2];
    SessionState sessions[2]{};
    QsaState states[2]{};
    for (int i = 0; i < 2; ++i) {
        assert(tables[i].allocate(8) == cudaSuccess);
        sessions[i].qsa_states = &states[i];
        sessions[i].qsa_alloc = 1;
        states[i].shared_residency = true;
        states[i].n_pages = 8;
        states[i].host.fp16_host_pages = tables[i].device_pages();
    }
    const SharedKvTable::Stage stages[] = {{0, tables[0], sessions[0], nullptr},
                                           {1, tables[1], sessions[1], nullptr}};
    assert(SharedKvTable::publish_extent(pool, lease, 2, stages) == cudaSuccess);
    assert(SharedKvTable::covers_prefix(lease, stages, 2));
    assert(!SharedKvTable::covers_prefix(lease, stages, 3));
    fail_peer_copy = true;
    assert(SharedKvTable::publish_extent(pool, lease, 3, stages) == cudaErrorUnknown);
    assert(current_device == 0 && pool.available() == 2);
    assert(tables[0].published() == 3 && tables[1].published() == 2);
    assert(!SharedKvTable::covers_prefix(lease, stages, 3));
    fail_peer_copy = false;
    assert(SharedKvTable::publish_extent(pool, lease, 3, stages) == cudaSuccess);
    for (const auto& state : states)
        assert(state.host_page_count == 3 && state.host_page_ids == lease.pages().data());
    assert(SharedKvTable::retire(pool, lease, stages) == cudaErrorUnknown);
    assert(current_device == 0 && pool.available() == 2 && !pool.reserve(other, 3));
    assert(tables[0].published() == 0 && tables[1].published() == 3);
    assert(!SharedKvTable::covers_prefix(lease, stages, 1));
    fail_peer_sync = false;
    assert(SharedKvTable::retire(pool, lease, stages) == cudaSuccess);
    assert(pool.available() == 8 && current_device == 0);
    assert(!states[0].host_page_ids && !states[1].host_page_ids);
    assert(invalidations[0] == 1 && invalidations[1] == 2);
    assert(pool.reserve(lease, 6));
    assert(SharedKvTable::publish_extent(pool, lease, 3, stages) == cudaSuccess);
    fail_peer_sync = true;
    assert(SharedKvTable::park_prefix(pool, lease, 2, stages) == cudaErrorUnknown);
    assert(pool.available() == 2 && lease.pages().size() == 3);
    assert(tables[0].published() == 3 && tables[1].published() == 3);
    fail_peer_sync = false;
    assert(SharedKvTable::park_prefix(pool, lease, 2, stages) == cudaSuccess);
    assert(SharedKvTable::covers_prefix(lease, stages, 2));
    assert(!SharedKvTable::covers_prefix(lease, stages, 3));
    states[0].host_page_ids = nullptr;
    assert(!SharedKvTable::covers_prefix(lease, stages, 2));
    states[0].host_page_ids = lease.pages().data();
    assert(pool.available() == 6 && states[0].host_page_count == 2 && states[1].host_page_count == 2);
    assert(pool.reserve(other, 1) && pool.materialize(other, 1) && other.pages()[0] == 2);
    pool.release(other);
    assert(SharedKvTable::retire(pool, lease, stages) == cudaSuccess && pool.available() == 8);
}
