#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core { class DeviceArena; }

namespace basalt::qwen4exp {

struct SwiftLoraSite {
    std::string name;
    uint64_t inputs;
    uint64_t outputs;
};

class SwiftLora {
public:
    SwiftLora(const std::string& path, const std::vector<SwiftLoraSite>& sites);
    std::size_t factor_bytes() const { return bytes_; }
    uint64_t state_identity(uint64_t seed) const;

private:
    friend class SwiftLoraDevice;
    struct Factors { std::vector<uint16_t> a, b; };
    std::map<std::string, Factors> factors_;
    std::size_t bytes_ = 0;
};

struct SwiftLoraFactors {
    const uint16_t* a;
    const uint16_t* b;
};

class SwiftLoraDevice {
public:
    SwiftLoraDevice(const SwiftLora& adapter, int device);
    ~SwiftLoraDevice();
    SwiftLoraDevice(const SwiftLoraDevice&) = delete;
    SwiftLoraDevice& operator=(const SwiftLoraDevice&) = delete;

    const SwiftLoraFactors* find(const std::string& name) const;
    const float* scale() const { return scale_; }
    void set_scale(float scale);

private:
    int device_;
    std::unique_ptr<core::DeviceArena> arena_;
    std::map<std::string, SwiftLoraFactors> factors_;
    float* scale_ = nullptr;
};

}
