#include "basalt/models/qwen4exp/swift_lora.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/conversation_store.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/kernels/swift_lora.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace basalt::qwen4exp {
namespace {

bool matches(const GgufFile& file, const char* key, const char* value) {
    const auto* meta = file.get(key);
    return meta && meta->type == MetaType::STRING && meta->s == value;
}

std::vector<uint16_t> read_factor(const GgufFile& file, const std::string& name,
                                  const std::vector<uint64_t>& shape) {
    constexpr int bf16 = 30;
    const auto* tensor = file.find(name);
    if (!tensor || tensor->type != bf16 || tensor->shape != shape ||
        file.data_start() > file.file_size() || tensor->offset > file.file_size() - file.data_start() ||
        tensor->elements() > (file.file_size() - file.data_start() - tensor->offset) / sizeof(uint16_t))
        throw std::runtime_error("Swift LoRA: invalid factor " + name);
    std::vector<uint16_t> bits(tensor->elements());
    std::memcpy(bits.data(), file.tensor_data(*tensor), bits.size() * sizeof(uint16_t));
    if (std::any_of(bits.begin(), bits.end(), [](uint16_t value) { return (value & 0x7f80) == 0x7f80; }))
        throw std::runtime_error("Swift LoRA: nonfinite factor " + name);
    return bits;
}

}

SwiftLora::SwiftLora(const std::string& path, const std::vector<SwiftLoraSite>& sites) {
    constexpr uint64_t rank = kernels::swift_lora_rank;
    const GgufFile file(path);
    const auto* alpha = file.get("adapter.lora_alpha");
    if (!matches(file, "general.architecture", "qwen4exp") || !matches(file, "general.type", "adapter") ||
        !matches(file, "adapter.type", "lora") || !alpha || alpha->type != MetaType::F32 || alpha->f != static_cast<float>(rank))
        throw std::runtime_error("Swift LoRA: expected qwen4exp rank-32 adapter with alpha 32");
    if (sites.size() != 300 || file.tensors().size() != 2 * sites.size())
        throw std::runtime_error("Swift LoRA: incomplete projection set");
    for (const auto& site : sites) {
        if (!site.inputs || !site.outputs || factors_.contains(site.name))
            throw std::runtime_error("Swift LoRA: invalid base projection " + site.name);
        auto a = read_factor(file, site.name + ".lora_a", {site.inputs, rank});
        auto b = read_factor(file, site.name + ".lora_b", {rank, site.outputs});
        bytes_ += (a.size() + b.size()) * sizeof(uint16_t);
        factors_.emplace(site.name, Factors{std::move(a), std::move(b)});
    }
}

uint64_t SwiftLora::state_identity(uint64_t seed) const {
    using core::conversation_store_hash;
    for (const auto& [name, factors] : factors_) {
        seed = conversation_store_hash(name.data(), name.size(), seed);
        seed = conversation_store_hash(factors.a.data(), factors.a.size() * sizeof(uint16_t), seed);
        seed = conversation_store_hash(factors.b.data(), factors.b.size() * sizeof(uint16_t), seed);
    }
    return seed;
}

SwiftLoraDevice::SwiftLoraDevice(const SwiftLora& adapter, int device) : device_(device) {
    const core::OnDevice on(device_);
    arena_ = std::make_unique<core::DeviceArena>(adapter.factor_bytes() + 512 * adapter.factors_.size() + 256, device_);
    auto upload = [&](const std::vector<uint16_t>& values) {
        const auto bytes = values.size() * sizeof(uint16_t);
        auto* dest = static_cast<uint16_t*>(arena_->alloc(bytes));
        const auto status = cudaMemcpy(dest, values.data(), bytes, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return dest;
    };
    for (const auto& [name, factors] : adapter.factors_)
        factors_.emplace(name, SwiftLoraFactors{upload(factors.a), upload(factors.b)});
    scale_ = static_cast<float*>(arena_->alloc(sizeof(float)));
    set_scale(0.0f);
}

SwiftLoraDevice::~SwiftLoraDevice() {
    const core::OnDevice on(device_);
    arena_.reset();
}

void SwiftLoraDevice::set_scale(float scale) {
    if (!std::isfinite(scale)) throw std::invalid_argument("Swift LoRA: nonfinite scale");
    const core::OnDevice on(device_);
    const auto status = cudaMemcpy(scale_, &scale, sizeof(scale), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

const SwiftLoraFactors* SwiftLoraDevice::find(const std::string& name) const {
    const auto entry = factors_.find(name);
    return entry != factors_.end() ? &entry->second : nullptr;
}

}
