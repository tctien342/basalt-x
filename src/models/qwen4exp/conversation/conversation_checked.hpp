#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

namespace basalt::core {
struct ConversationKv;
struct QsaState;
struct ModelGeometry;
}

namespace basalt::core::conversation_detail {
bool trim_kv(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
             int64_t upto, bool index, std::string& error);
bool restore_kv(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                int64_t upto, bool index, bool mapped_host, std::string& error);

inline bool add(size_t& total, size_t n) {
    if (n > std::numeric_limits<size_t>::max() - total) return false;
    total += n;
    return true;
}
inline bool product(size_t& out, std::initializer_list<uint64_t> factors) {
    size_t value = 1;
    for (uint64_t factor : factors) {
        if (factor > std::numeric_limits<size_t>::max() ||
            (factor && value > std::numeric_limits<size_t>::max() / factor)) return false;
        value *= (size_t) factor;
    }
    out = value;
    return true;
}
} // namespace basalt::core::conversation_detail
