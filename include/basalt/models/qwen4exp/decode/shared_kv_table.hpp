#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <cuda_runtime_api.h>

#include "basalt/models/qwen4exp/decode/shared_kv_pages.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "basalt/models/qwen4exp/kernels/kv_stream.hpp"

namespace basalt::core {

// One stable table per CUDA device used by a sequence. The scheduler owns its
// lifetime and drains all streams that use it before publishing or releasing.
class SharedKvTable {
public:
    SharedKvTable() = default;
    SharedKvTable(const SharedKvTable&) = delete;
    SharedKvTable& operator=(const SharedKvTable&) = delete;
    ~SharedKvTable() {
        if (pages_) cudaFree(pages_);
    }

    cudaError_t allocate(uint32_t capacity) {
        if (pages_ || !capacity) return cudaErrorInvalidValue;
        const auto status = cudaMalloc(reinterpret_cast<void**>(&pages_), size_t(capacity) * sizeof(*pages_));
        if (status == cudaSuccess) capacity_ = capacity;
        return status;
    }

    struct Stage {
        int device;
        SharedKvTable& table;
        SessionState& session;
        cudaStream_t stream;
    };

    // A checkpoint is a source only while all of its pages are published in every stage.
    static bool covers_prefix(const SharedKvPages::Lease& lease, std::span<const Stage> stages, uint32_t pages) {
        const auto ids = lease.pages();
        if (!pages || pages > ids.size() || stages.empty()) return false;
        for (const auto& stage : stages) {
            const auto& session = stage.session;
            if (stage.table.published_owner_ != &lease || stage.table.published_ < pages ||
                !session.qsa_states || session.qsa_alloc <= 0)
                return false;
            for (int64_t i = 0; i < session.qsa_alloc; ++i) {
                const auto& state = session.qsa_states[session.qsa_ord0 + i];
                if (!state.shared_residency || state.host.fp16_host_pages != stage.table.pages_ ||
                    state.host_page_ids != ids.data() || state.host_page_count < pages)
                    return false;
            }
        }
        return true;
    }

    // Do not dispatch work until every stage has published the requested extent.
    // On failure keep the lease and successful publications for retry or retirement.
    static cudaError_t publish_extent(SharedKvPages& pool, SharedKvPages::Lease& lease,
                                      uint32_t pages, std::span<const Stage> stages) {
        if (!pages || stages.empty()) return cudaErrorInvalidValue;
        int original = 0;
        auto status = cudaGetDevice(&original);
        if (status != cudaSuccess) return status;
        for (const auto& stage : stages) {
            if (!stage.table.pages_ || pages > stage.table.capacity_ ||
                (stage.table.published_owner_ && stage.table.published_owner_ != &lease))
                return cudaErrorInvalidValue;
        }
        if (!pool.materialize(lease, pages)) return cudaErrorInvalidValue;
        for (const auto& stage : stages) {
            status = cudaSetDevice(stage.device);
            if (status == cudaSuccess) status = stage.table.publish(stage.session, lease);
            if (status != cudaSuccess) break;
        }
        const auto restored = cudaSetDevice(original);
        return status == cudaSuccess ? restored : status;
    }

    // Stage streams must join all work that can touch this lease, including
    // speculative and peer prefill work. A failure keeps the lease reserved.
    static cudaError_t retire(SharedKvPages& pool, SharedKvPages::Lease& lease, std::span<const Stage> stages) {
        if (!lease.pages().empty() && stages.empty()) return cudaErrorInvalidValue;
        int original = 0;
        auto status = cudaGetDevice(&original);
        if (status != cudaSuccess) return status;
        for (const auto& stage : stages) {
            if (stage.table.published_owner_ && stage.table.published_owner_ != &lease)
                return cudaErrorInvalidValue;
        }
        for (const auto& stage : stages) {
            status = cudaSetDevice(stage.device);
            if (status == cudaSuccess) status = stage.table.invalidate(stage.session, stage.stream);
            if (status != cudaSuccess) break;
        }
        const auto restored = cudaSetDevice(original);
        if (status != cudaSuccess) return status;
        if (restored != cudaSuccess) return restored;
        pool.release(lease);
        return cudaSuccess;
    }

    // Shrink a completed request to its cached prefix once every device has
    // invalidated the discarded range. A failed fence leaves capacity charged.
    static cudaError_t park_prefix(SharedKvPages& pool, SharedKvPages::Lease& lease,
                                   uint32_t keep, std::span<const Stage> stages) {
        if (!keep || keep > lease.pages().size() || stages.empty()) return cudaErrorInvalidValue;
        int original = 0;
        auto status = cudaGetDevice(&original);
        if (status != cudaSuccess) return status;
        for (const auto& stage : stages) {
            if (stage.table.published_owner_ != &lease || stage.table.published_ != lease.pages().size() ||
                !stage.session.qsa_states || stage.session.qsa_alloc <= 0)
                return cudaErrorInvalidValue;
            for (int64_t i = 0; i < stage.session.qsa_alloc; ++i) {
                const auto& st = stage.session.qsa_states[stage.session.qsa_ord0 + i];
                if (!st.shared_residency || st.host.fp16_host_pages != stage.table.pages_)
                    return cudaErrorInvalidValue;
            }
        }
        const auto drop = lease.pages().size() - keep;
        for (const auto& stage : stages) {
            status = cudaSetDevice(stage.device);
            if (status != cudaSuccess) break;
            for (int64_t i = 0; i < stage.session.qsa_alloc; ++i) {
                const auto& st = stage.session.qsa_states[stage.session.qsa_ord0 + i];
                basalt::kernels::kv_stream_invalidate(st.map, stage.table.pages_ + keep, drop, stage.stream);
            }
            status = cudaStreamSynchronize(stage.stream);
            if (status != cudaSuccess) break;
        }
        const auto restored = cudaSetDevice(original);
        if (status != cudaSuccess) return status;
        if (restored != cudaSuccess) return restored;
        if (!pool.trim(lease, keep)) return cudaErrorInvalidValue;
        for (const auto& stage : stages) {
            stage.table.published_ = keep;
            for (int64_t i = 0; i < stage.session.qsa_alloc; ++i)
                stage.session.qsa_states[stage.session.qsa_ord0 + i].host_page_count = keep;
        }
        return cudaSuccess;
    }

    const int32_t* device_pages() const { return pages_; }
    uint32_t published() const { return published_; }

    // The existing prefix stays resident and keeps its device table entries.
    cudaError_t publish(const SharedKvPages::Lease& lease) {
        const auto ids = lease.pages();
        if (ids.size() > capacity_ || ids.size() < published_) return cudaErrorInvalidValue;
        if (published_owner_ && published_owner_ != &lease) return cudaErrorInvalidValue;
        const size_t fresh = ids.size() - published_;
        if (!fresh) return cudaSuccess;
        const auto status = cudaMemcpy(pages_ + published_, ids.data() + published_,
                                       fresh * sizeof(*pages_), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            published_ = static_cast<uint32_t>(ids.size());
            published_owner_ = &lease;
        }
        return status;
    }

    // Validate all layers before publishing. The CPU mirror follows the lease;
    // graph-captured device pointers remain fixed across publications.
    cudaError_t publish(SessionState& session, const SharedKvPages::Lease& lease) {
        if (!session.qsa_states || session.qsa_alloc <= 0 || !pages_) return cudaErrorInvalidValue;
        for (int64_t i = 0; i < session.qsa_alloc; ++i) {
            const auto& state = session.qsa_states[session.qsa_ord0 + i];
            if (!state.shared_residency || state.host.fp16_host_pages != pages_ ||
                lease.pages().size() > static_cast<size_t>(state.n_pages))
                return cudaErrorInvalidValue;
        }
        const auto status = publish(lease);
        if (status != cudaSuccess) return status;
        for (int64_t i = 0; i < session.qsa_alloc; ++i) {
            auto& state = session.qsa_states[session.qsa_ord0 + i];
            state.host_page_ids = lease.pages().data();
            state.host_page_count = published_;
        }
        return cudaSuccess;
    }

    cudaError_t invalidate(SessionState& session, cudaStream_t stream) {
        if (!published_) return cudaSuccess;
        if (!session.qsa_states || session.qsa_alloc <= 0) return cudaErrorInvalidValue;
        std::vector<basalt::kernels::KvStreamMap> maps;
        maps.reserve(static_cast<size_t>(session.qsa_alloc));
        for (int64_t i = 0; i < session.qsa_alloc; ++i) {
            const auto& state = session.qsa_states[session.qsa_ord0 + i];
            if (!state.shared_residency || state.host.fp16_host_pages != pages_)
                return cudaErrorInvalidValue;
            maps.push_back(state.map);
        }
        const auto status = invalidate(maps, stream);
        if (status != cudaSuccess) return status;
        for (int64_t i = 0; i < session.qsa_alloc; ++i) {
            auto& state = session.qsa_states[session.qsa_ord0 + i];
            state.host_page_ids = nullptr;
            state.host_page_count = 0;
        }
        return cudaSuccess;
    }

    // Call on each owning device before releasing a lease to the allocator.
    // Callers order the stream after all readers and writers of these pages.
    cudaError_t invalidate(std::span<const basalt::kernels::KvStreamMap> maps, cudaStream_t stream) {
        if (!published_) return cudaSuccess;
        for (const auto& map : maps) basalt::kernels::kv_stream_invalidate(map, pages_, published_, stream);
        const auto status = cudaStreamSynchronize(stream);
        if (status == cudaSuccess) {
            published_ = 0;
            published_owner_ = nullptr;
        }
        return status;
    }

private:
    int32_t* pages_ = nullptr;
    uint32_t capacity_ = 0, published_ = 0;
    const SharedKvPages::Lease* published_owner_ = nullptr;
};

}  // namespace basalt::core
