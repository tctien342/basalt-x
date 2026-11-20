#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>

namespace basalt::core {

// Experimental transports use the driver peer access enabled by the GPU selector.
inline bool direct_peer_enabled(int dst, int src) {
    static const bool enabled = [] {
        const char* v = std::getenv("BASALT_DIRECT_PEER");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    int access = 0;
    return enabled && src != dst && cudaDeviceCanAccessPeer(&access, dst, src) == cudaSuccess && access;
}

template <class Fn> Fn direct_peer_entry(const char* name) {
    void* entry = nullptr;
    cudaDriverEntryPointQueryResult result{};
    if (cudaGetDriverEntryPointByVersion(name, &entry, 12000, cudaEnableDefault, &result) != cudaSuccess ||
        result != cudaDriverEntryPointSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return reinterpret_cast<Fn>(entry);
}

inline bool direct_peer_grant(void* base, size_t bytes, int accessor) {
    static const auto set_access = direct_peer_entry<decltype(&cuMemSetAccess)>("cuMemSetAccess");
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = accessor;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return set_access != nullptr && bytes > 0 &&
           set_access((CUdeviceptr) base, bytes, &access, 1) == CUDA_SUCCESS;
}

// Legacy cudaMalloc allocations inherit the selector's peer mappings. A loan
// from a VMM arena needs an explicit grant covering the requested range.
inline bool direct_peer_pointer(const void* p, int device, int accessor = -1, size_t bytes = 0) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (a.type != cudaMemoryTypeDevice || a.device != device) return false;
    static const auto attribute = direct_peer_entry<decltype(&cuPointerGetAttribute)>("cuPointerGetAttribute");
    unsigned int legacy = 0;
    if (attribute != nullptr &&
        attribute(&legacy, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, (CUdeviceptr) p) == CUDA_SUCCESS && legacy)
        return true;
    if (accessor < 0 || bytes == 0) return false;
    static const auto get_access = direct_peer_entry<decltype(&cuMemGetAccess)>("cuMemGetAccess");
    if (get_access == nullptr) return false;
    CUmemLocation location{};
    location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    location.id = accessor;
    unsigned long long first = 0, last = 0;
    const auto start = (CUdeviceptr) p;
    return get_access(&first, &location, start) == CUDA_SUCCESS &&
           get_access(&last, &location, start + bytes - 1) == CUDA_SUCCESS &&
           first == CU_MEM_ACCESS_FLAGS_PROT_READWRITE && last == CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

}  // namespace basalt::core
