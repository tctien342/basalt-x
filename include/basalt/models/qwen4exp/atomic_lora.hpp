#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core { class DeviceArena; }

namespace basalt::qwen4exp {

struct AtomicLoraSite {
    std::string name;
    std::size_t input;
    std::size_t output;
    std::size_t experts = 1;
    bool embedding = false;
};

/// Owns compact host FP32 factors for one validated Flash-Next rank-one adapter.
/// The expected sites and dimensions come from the base pack.
class AtomicLora {
public:
    AtomicLora(const std::string& path, const std::vector<AtomicLoraSite>& sites);

    /// One A row per expert. For the embedding, this is the vector indexed by token ID.
    const std::vector<float>& input_factor(const std::string& name) const { return inputs_.at(name); }
    const std::vector<float>& output_factor() const { return output_; }
    std::size_t factor_bytes() const;
    uint64_t state_identity(uint64_t seed) const;

private:
    friend class AtomicLoraDevice;
    std::map<std::string, std::vector<float>> inputs_;
    std::vector<float> output_;
};

/// One immutable factor upload and a graph-stable request scale on a CUDA device.
class AtomicLoraDevice {
public:
    AtomicLoraDevice(const AtomicLora& adapter, int device);
    ~AtomicLoraDevice();
    AtomicLoraDevice(const AtomicLoraDevice&) = delete;
    AtomicLoraDevice& operator=(const AtomicLoraDevice&) = delete;

    const float* input_factor(const std::string& name) const { return inputs_.at(name); }
    const float* find_input_factor(const std::string& name) const {
        const auto it = inputs_.find(name);
        return it != inputs_.end() ? it->second : nullptr;
    }
    const float* output_factor() const { return output_; }
    const float* scale() const { return scale_; }
    const float* const* routed_factors() const { return routed_; }

    /// The caller drains prior adapter consumers on this device before changing the scale.
    void set_scale(float value);

private:
    int device_;
    std::unique_ptr<core::DeviceArena> arena_;
    std::map<std::string, const float*> inputs_;
    const float* output_ = nullptr;
    float* scale_ = nullptr;
    const float** routed_ = nullptr;
};

}
