#include "basalt/core/conversation_store.hpp"
#include "basalt/models/qwen4exp/atomic_lora.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/on_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>

namespace basalt::qwen4exp {
namespace {

bool matches(const GgufFile& file, const char* key, const char* value) {
    const auto* meta = file.get(key);
    return meta && meta->type == MetaType::STRING && meta->s == value;
}

const uint8_t* factor(const GgufFile& file, const TensorInfo& tensor,
                      const std::vector<uint64_t>& shape) {
    const auto start = file.data_start();
    const auto size = file.file_size();
    if (tensor.type != 0 || tensor.shape != shape || start > size ||
        tensor.offset > size - start || tensor.elements() > (size - start - tensor.offset) / sizeof(float))
        throw std::runtime_error("AtomicChat LoRA: invalid factor " + tensor.name);
    return file.tensor_data(tensor);
}

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

}

AtomicLora::AtomicLora(const std::string& path, const std::vector<AtomicLoraSite>& sites) {
    const GgufFile file(path);
    const auto* alpha = file.get("adapter.lora_alpha");
    if (!matches(file, "general.architecture", "qwen4exp") ||
        !matches(file, "general.type", "adapter") || !matches(file, "adapter.type", "lora") ||
        !alpha || alpha->type != MetaType::F32 || alpha->f != 1.0)
        throw std::runtime_error("AtomicChat LoRA: expected qwen4exp rank-one adapter with alpha 1");
    if (sites.size() != 146 || file.tensors().size() != 2 * sites.size())
        throw std::runtime_error("AtomicChat LoRA: incomplete writer set");

    std::map<std::string, const TensorInfo*> tensors;
    for (const auto& tensor : file.tensors()) {
        if (!tensors.emplace(tensor.name, &tensor).second)
            throw std::runtime_error("AtomicChat LoRA: duplicate factor " + tensor.name);
    }
    std::set<std::string> names;
    for (const auto& site : sites) {
        if (!names.insert(site.name).second || !site.input || !site.output || !site.experts)
            throw std::runtime_error("AtomicChat LoRA: invalid base writer " + site.name);
        const auto a = tensors.find(site.name + ".lora_a");
        const auto b = tensors.find(site.name + ".lora_b");
        if (a == tensors.end() || b == tensors.end())
            throw std::runtime_error("AtomicChat LoRA: missing factor for " + site.name);
        const std::size_t width = site.embedding ? site.input : site.output;
        std::vector<uint64_t> a_shape = site.embedding ? std::vector<uint64_t>{1, site.output}
                                                       : std::vector<uint64_t>{site.input, 1};
        std::vector<uint64_t> b_shape = {1, width};
        if (site.experts > 1) {
            a_shape.push_back(site.experts);
            b_shape.push_back(site.experts);
        }
        const auto* a_bytes = factor(file, *a->second, a_shape);
        const auto* b_bytes = factor(file, *b->second, b_shape);
        const auto stride = width * sizeof(float);
        if (output_.empty()) {
            output_.resize(width);
            std::memcpy(output_.data(), b_bytes, stride);
            if (!finite(output_))
                throw std::runtime_error("AtomicChat LoRA: non-finite output factor");
        }
        for (std::size_t expert = 0; expert < site.experts; ++expert) {
            if (output_.size() != width || std::memcmp(output_.data(), b_bytes + expert * stride, stride))
                throw std::runtime_error("AtomicChat LoRA: different output factor for " + site.name);
        }
        auto& input = inputs_[site.name];
        input.resize(a->second->elements());
        std::memcpy(input.data(), a_bytes, input.size() * sizeof(float));
        if (!finite(input))
            throw std::runtime_error("AtomicChat LoRA: non-finite input factor for " + site.name);
    }
}

uint64_t AtomicLora::state_identity(uint64_t seed) const {
    using core::conversation_store_hash;
    uint64_t identity = conversation_store_hash(output_.data(), output_.size() * sizeof(float), seed);
    for (const auto& [name, input] : inputs_) {
        identity = conversation_store_hash(name.data(), name.size(), identity);
        identity = conversation_store_hash(input.data(), input.size() * sizeof(float), identity);
    }
    return identity;
}

std::size_t AtomicLora::factor_bytes() const {
    std::size_t count = output_.size();
    for (const auto& entry : inputs_) count += entry.second.size();
    return count * sizeof(float);
}

AtomicLoraDevice::AtomicLoraDevice(const AtomicLora& adapter, int device)
    : device_(device) {
    const core::OnDevice on(device_);
    arena_ = std::make_unique<core::DeviceArena>(
        adapter.factor_bytes() + 256 * (adapter.inputs_.size() + 3) + 48 * sizeof(float*), device_);
    auto upload = [&](const std::vector<float>& values) {
        const auto bytes = values.size() * sizeof(float);
        auto* dest = static_cast<float*>(arena_->alloc(bytes));
        const auto status = cudaMemcpy(dest, values.data(), bytes, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return dest;
    };
    output_ = upload(adapter.output_);
    for (const auto& entry : adapter.inputs_)
        inputs_.emplace(entry.first, upload(entry.second));
    const float* routed[48];
    for (int l = 0; l < 48; ++l)
        routed[l] = input_factor("blk." + std::to_string(l) + ".ffn_down_exps.weight");
    routed_ = static_cast<const float**>(arena_->alloc(sizeof(routed)));
    const auto status = cudaMemcpy(routed_, routed, sizeof(routed), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    scale_ = static_cast<float*>(arena_->alloc(sizeof(float)));
    set_scale(0);
}

AtomicLoraDevice::~AtomicLoraDevice() {
    const core::OnDevice on(device_);
    arena_.reset();
}

void AtomicLoraDevice::set_scale(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("AtomicChat LoRA: non-finite request scale");
    const core::OnDevice on(device_);
    const auto status = cudaMemcpy(scale_, &value, sizeof(value), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}
