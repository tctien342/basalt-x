// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "basalt/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace basalt::core {

/// Byte storage for a checkpoint's running state.  Large blocks go back to a small shared pool instead of the OS and
/// are handed out again without being zeroed: a fresh block of this size costs more in page faults and zero-fill
/// than the device copy into it, and a checkpoint is saved on the request path.  A server reserves the blocks its
/// checkpoints take at their most before its RAM tier is sized, and the pool keeps that many through a RAM shed.
template <class T>
struct PooledAlloc {
    using value_type = T;
    static constexpr size_t kMinBytes = size_t(1) << 20;   // smaller blocks are not worth pooling
    static constexpr size_t kMaxBlocks = 4;                // blocks kept beyond the reserved ones; the rest are freed
    PooledAlloc() = default;
    template <class U>
    PooledAlloc(const PooledAlloc<U>&) noexcept {}
    T* allocate(size_t n) {
        const size_t bytes = n * sizeof(T);
        if (bytes >= kMinBytes) {
            Pool& p = pool();
            std::lock_guard<std::mutex> lk(p.mu);
            for (size_t i = 0; i < p.blocks.size(); ++i)
                if (p.blocks[i].second == bytes) {
                    T* r = static_cast<T*>(p.blocks[i].first);
                    p.blocks.erase(p.blocks.begin() + (std::ptrdiff_t) i);
                    return r;
                }
        }
        return static_cast<T*>(::operator new(bytes));
    }
    void deallocate(T* ptr, size_t n) noexcept {
        const size_t bytes = n * sizeof(T);
        if (bytes >= kMinBytes) {
            Pool& p = pool();
            std::lock_guard<std::mutex> lk(p.mu);
            if (p.blocks.size() < kMaxBlocks + p.reserved) {
                p.blocks.emplace_back(ptr, bytes);
                return;
            }
        }
        ::operator delete(ptr);
    }
    template <class U>
    void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }   // resize leaves new bytes as they are
    template <class U, class... A>
    void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
    /// Allocates `count` blocks of `bytes` now, their pages written, for checkpoints to take.  False when they cannot
    /// be allocated (none is kept then).
    static bool reserve(size_t bytes, size_t count) {
        if (bytes < kMinBytes || count == 0) return true;
        std::vector<void*> got;
        for (size_t i = 0; i < count; ++i) {
            void* b = ::operator new(bytes, std::nothrow);
            if (b == nullptr) {
                for (void* x : got) ::operator delete(x);
                return false;
            }
            std::memset(b, 0, bytes);
            got.push_back(b);
        }
        Pool& p = pool();
        std::lock_guard<std::mutex> lk(p.mu);
        for (void* b : got) p.blocks.emplace_back(b, bytes);
        p.reserved += count;
        return true;
    }
    /// Frees the blocks the pool keeps beyond the reserved ones; returns their bytes.
    static size_t release_pooled() {
        Pool& p = pool();
        std::lock_guard<std::mutex> lk(p.mu);
        size_t n = 0;
        while (p.blocks.size() > p.reserved) {
            n += p.blocks.back().second;
            ::operator delete(p.blocks.back().first);
            p.blocks.pop_back();
        }
        return n;
    }
    template <class U>
    bool operator==(const PooledAlloc<U>&) const noexcept { return true; }
    template <class U>
    bool operator!=(const PooledAlloc<U>&) const noexcept { return false; }

private:
    struct Pool {
        std::mutex mu;
        std::vector<std::pair<void*, size_t>> blocks;
        size_t reserved = 0;
    };
    static Pool& pool() {
        static Pool* p = new Pool;   // never destroyed: checkpoints may outlive static destruction
        return *p;
    }
};
class CheckpointBytes {
    using Storage = std::vector<uint8_t, PooledAlloc<uint8_t>>;

public:
    using allocator_type = Storage::allocator_type;

    CheckpointBytes() = default;
    CheckpointBytes(size_t n, uint8_t value) : owned_(n, value) {}
    CheckpointBytes(const CheckpointBytes& other) : owned_(other.storage()) {}
    CheckpointBytes& operator=(const CheckpointBytes& other) {
        if (this != &other) *this = CheckpointBytes(other);
        return *this;
    }
    CheckpointBytes(CheckpointBytes&&) noexcept = default;
    CheckpointBytes& operator=(CheckpointBytes&&) noexcept = default;

    // Only completed host snapshots can share. Ordinary copies keep their
    // independent backing. A mutable accessor detaches before returning.
    CheckpointBytes shared_copy() {
        CheckpointBytes out;
        if (empty()) return out;
        if (!shared_) shared_ = std::make_shared<Storage>(std::move(owned_));
        out.shared_ = shared_;
        return out;
    }

    size_t size() const { return storage().size(); }
    size_t capacity() const { return storage().capacity(); }
    bool empty() const { return storage().empty(); }
    const uint8_t* data() const { return storage().data(); }
    uint8_t* data() { return writable().data(); }
    auto begin() const { return storage().begin(); }
    auto end() const { return storage().end(); }
    auto begin() { return writable().begin(); }
    auto end() { return writable().end(); }
    const uint8_t& operator[](size_t i) const { return storage()[i]; }
    uint8_t& operator[](size_t i) { return writable()[i]; }
    bool operator==(const CheckpointBytes& other) const { return storage() == other.storage(); }
    void resize(size_t n) { writable().resize(n); }
    void resize(size_t n, uint8_t value) { writable().resize(n, value); }
    void assign(size_t n, uint8_t value) { writable().assign(n, value); }
    void push_back(uint8_t value) { writable().push_back(value); }
    void pop_back() { writable().pop_back(); }

private:
    const Storage& storage() const { return shared_ ? *shared_ : owned_; }
    Storage& writable() {
        if (shared_) {
            if (shared_.use_count() == 1) owned_ = std::move(*shared_);
            else owned_ = *shared_;
            shared_.reset();
        }
        return owned_;
    }

    Storage owned_;
    std::shared_ptr<Storage> shared_;
};

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    /// What took the checkpoint: the end of a prompt, a fixed stride inside a long one, or the end of the system prompt.
    enum Source : uint8_t { kTurnEnd = 0, kPeriodic = 1, kRoot = 2 };
    Source source = kTurnEnd;
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    /// The model's running state, its parts as the model names them.
    CheckpointBytes state_a, state_b, state_c, state_d, state_e;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    ConversationCheckpoint shared_copy() {
        ConversationCheckpoint out;
        out.source = source;
        out.ids = ids;
        out.imgs = imgs;
        out.state_a = state_a.shared_copy();
        out.state_b = state_b.shared_copy();
        out.state_c = state_c.shared_copy();
        out.state_d = state_d.shared_copy();
        out.state_e = state_e.shared_copy();
        out.used = used;
        out.stage_parts.reserve(stage_parts.size());
        for (auto& part : stage_parts) out.stage_parts.push_back(part.shared_copy());
        return out;
    }

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               state_a.capacity() + state_b.capacity() + state_c.capacity() + state_d.capacity() + state_e.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes(bool only_unique = false) const {
        return k.bytes(only_unique) + v.bytes(only_unique) + k_scale.bytes(only_unique) +
               v_scale.bytes(only_unique) + pooled.bytes(only_unique);
    }
    ConversationKv shared_copy() const {
        return {format, cells, heads, head_dim, page_size, pooled_rows, idx_dim,
                k.shared_copy(), v.shared_copy(), k_scale.shared_copy(), v_scale.shared_copy(), pooled.shared_copy()};
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes(bool only_unique = false) const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes(only_unique);
        for (const auto& s : stages) n += s.bytes(only_unique);
        return n;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    for (auto& c : checks) {
        const bool whole = c.stage_parts.size() == stages;
        out.complete.push_back(whole);
        if (!whole) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part = std::move(c.stage_parts[k]);
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.stage_parts.clear();
            out.parts[k].push_back(std::move(part));
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    /// an idle batch slot's conversation moved out of VRAM (ConversationCache::put_slot): outside the byte budget
    bool from_slot = false;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;

    // The recovery writer only accepts unsplit images. Give it independent
    // metadata while sharing the completed host payload with the live cache.
    std::optional<SavedConversation> shared_recovery_view() {
        if (!stage_images.empty() || !live.stage_parts.empty() ||
            std::any_of(checkpoints.begin(), checkpoints.end(),
                        [](const auto& checkpoint) { return !checkpoint.stage_parts.empty(); }))
            return std::nullopt;
        SavedConversation out;
        out.geometry = geometry;
        out.layer_lo = layer_lo;
        out.layer_hi = layer_hi;
        out.live = live.shared_copy();
        out.cvec = cvec;
        out.from_slot = from_slot;
        out.checkpoints.reserve(checkpoints.size());
        for (auto& checkpoint : checkpoints) out.checkpoints.push_back(checkpoint.shared_copy());
        out.kv.reserve(kv.size());
        for (const auto& layer : kv) out.kv.push_back(layer.shared_copy());
        return out;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

/// Element-by-element token equality, any token width.  Every caller only asks "equal or not", so this is the shape
/// the bulk path below has to agree with.
template<class Token>
bool conversation_ids_equal_linear(const int32_t* ids, size_t n, const Token* prompt) {
    for (size_t i = 0; i < n; ++i)
        if (!(ids[i] == prompt[i])) return false;
    return true;
}

/// The same answer as `conversation_ids_equal_linear`.  A key is `int32`, and a prompt is `int32` in the serving path,
/// so that case compares in bulk (`memcmp` over the slice); any other token width falls back to the element-wise loop.
template<class Token>
bool conversation_ids_equal(const int32_t* ids, size_t n, const Token* prompt) {
    if constexpr (std::is_same_v<Token, int32_t>) return std::memcmp(ids, prompt, n * sizeof(int32_t)) == 0;
    else return conversation_ids_equal_linear(ids, n, prompt);
}

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !conversation_ids_equal(c.ids.data(), n, prompt.data())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    ConversationCache(size_t budget, size_t slots) : budget_(budget), slots_(slots) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    size_t bytes() const { return bytes_ + retained_bytes(); }
    size_t size() const { return entries_.size(); }
    /// the idle slots' conversations among them (put_slot) and their bytes
    size_t slot_size() const { return slot_count_; }
    size_t slot_bytes() const { return slot_bytes_; }
    size_t evictions() const { return evictions_; }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    // `owner`: the session the buffers were restored into; only that session's own state may be captured with them.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}, const void* owner = nullptr) {
        clear_retained();
        reuse_owner_ = owner;
        donor_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (enabled() && candidate.bytes() <= budget_ - (bytes_ - slot_bytes_)) reuse_ = std::move(candidate);
    }
    bool can_retain_complete(size_t bytes) const {
        return enabled() && bytes <= budget_ - (bytes_ - slot_bytes_);
    }
    // The accepted writer has its own metadata and shares the backing bytes.
    // Allow full reuse only on the next command and charge it as one
    // retained allocation. Later use can still reuse just the K/V.
    void retain_complete(SavedConversation&& image, const void* owner, uint64_t next_dispatch) {
        const bool complete = can_retain_complete(image.bytes());
        const auto tokens = static_cast<int64_t>(image.live.ids.size());
        retain(std::exchange(image.kv, {}), tokens, {}, owner);
        if (complete && !reuse_.kv.empty()) {
            ready_.emplace(std::move(image));
            ready_dispatch_ = next_dispatch;
        }
    }
    std::optional<SavedConversation> take_complete(const void* owner, uint64_t dispatch,
                                                   const std::vector<int32_t>& ids,
                                                   const std::vector<ConversationImageKey>& imgs, bool cvec) {
        if (!ready_ || !owner || owner != reuse_owner_ || dispatch != ready_dispatch_ ||
            ready_->live.ids != ids || ready_->live.imgs != imgs || ready_->cvec != cvec) return std::nullopt;
        ready_->kv = std::move(reuse_.kv);
        auto image = std::exchange(ready_, std::nullopt);
        reuse_ = {};
        return image;
    }
    void limit_reuse(int64_t first_dirty) {
        ready_.reset();
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
        donor_.unchanged = std::min(donor_.unchanged, first_dirty);
        if (donor_.unchanged <= 0) donor_ = {};
    }
    // Buffers retained for another session (a batch slot's, or the one a slot's copy has since overwritten) hold
    // another conversation's K/V: they are dropped, never handed to this session's capture.
    void drop_reuse_unless(const void* owner) {
        if (owner != reuse_owner_) clear_retained();
        if (owner != donor_.owner) donor_ = {};
    }
    const ConversationKvReuse* retained_reuse(const void* owner) const {
        return owner && owner == reuse_owner_ && !reuse_.kv.empty() ? &reuse_ : nullptr;
    }
    ConversationKvReuse take_reuse(const void* owner = nullptr) {
        drop_reuse_unless(owner);
        ready_.reset();
        return std::exchange(reuse_, {});
    }
    size_t retained_bytes() const { return reuse_.bytes() + (ready_ ? ready_->bytes() : 0); }
    // Call after ordinary eviction. A donor still in the cache keeps its image
    // and every borrower detaches before writing any shared segment.
    ConversationKvReuse share_donor(const void* owner) const {
        if (spill_ || !owner || owner != donor_.owner || donor_.unchanged < 32768) return {};
        for (const auto& entry : entries_) {
            if (entry.live.ids.data() != donor_.ids) continue;
            ConversationKvReuse shared;
            shared.captured_tokens = (int64_t) entry.live.ids.size();
            shared.unchanged_tokens = donor_.unchanged;
            shared.kv.reserve(entry.kv.size());
            for (const auto& layer : entry.kv) shared.kv.push_back(layer.shared_copy());
            return shared;
        }
        return {};
    }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() - slot_count_ < slots_ && budgeted_bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        forget(out);
        return out;
    }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    // The idle slots' conversations are outside the budget: they stay.
    bool make_room(size_t incoming, size_t held = 0) {
        return make_room(incoming, held, nullptr, nullptr);
    }

    // The regular victim still leaves. Its payload can be reused only for the
    // active session that kept an unchanged prefix when that entry was parked.
    bool make_room(size_t incoming, size_t held, const void* owner, ConversationKvReuse* reuse) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (spill_ || !owner || owner != donor_.owner) reuse = nullptr;
        if (budgeted_bytes() > budget_ - held - incoming) clear_retained();
        while (entries_.size() - slot_count_ >= slots_ || bytes_ - slot_bytes_ > budget_ - held - incoming)
            if (!evict_oldest(false, reuse)) break;
        return true;
    }

    void remember_park(const void* owner, int64_t kept) {
        donor_ = {};
        if (spill_ || !owner || kept <= 0 || entries_.empty()) return;
        const auto& e = entries_.back();
        if (!e.from_slot && e.stage_images.empty() && !e.kv.empty() && kept <= (int64_t) e.live.ids.size())
            donor_ = {e.live.ids.data(), owner, kept};
    }

    /// An idle batch slot's conversation, out of VRAM: kept whatever the byte budget (the caller has admitted its
    /// RAM), at most `keep` of them; the longest parked leaves first (to the tier below when spilling).
    void put_slot(SavedConversation&& image, size_t keep) {
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        while (slot_count_ > 0 && slot_count_ >= keep && evict_oldest(true)) {}
        image.from_slot = true;
        const size_t n = image.bytes();
        entries_.push_back(std::move(image));
        bytes_ += n;
        slot_bytes_ += n;
        ++slot_count_;
    }

    /// With spilling on, what make_room evicts waits in take_spilled() for the tier below (the disk) instead of going.
    void set_spill(bool on) { spill_ = on; }
    std::vector<SavedConversation> take_spilled() { return std::exchange(spilled_, {}); }

    // Drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest)) {
                SavedConversation gone = std::move(entries_[i]);
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                forget(gone);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }

    // Drop every parked conversation and the retained K/V buffers, to give their RAM back; returns their bytes.  With
    // spilling on, the conversations wait in take_spilled() for the tier below, as make_room's evictions do.
    size_t clear() {
        const size_t freed = bytes();
        if (spill_)
            for (auto& e : entries_) spilled_.push_back(std::move(e));
        entries_.clear();
        bytes_ = 0;
        slot_bytes_ = 0;
        slot_count_ = 0;
        clear_retained();
        reuse_owner_ = nullptr;
        donor_ = {};
        return freed;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        image.from_slot = false;
        entries_.push_back(std::move(image));
        bytes_ += n;
        return true;
    }

private:
    size_t budgeted_bytes() const { return bytes_ - slot_bytes_ + retained_bytes(); }
    void clear_retained() {
        ready_.reset();
        reuse_ = {};
    }
    // the bookkeeping of an entry that has left `entries_`
    void forget(const SavedConversation& e) {
        if (e.live.ids.data() == donor_.ids) donor_ = {};
        const size_t n = e.bytes();
        bytes_ -= n;
        if (e.from_slot) { slot_bytes_ -= n; --slot_count_; }
    }
    // the oldest entry of one kind (an idle slot's or a budgeted one) leaves; false when there is none
    bool evict_oldest(bool from_slot, ConversationKvReuse* reuse = nullptr) {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].from_slot != from_slot) continue;
            SavedConversation gone = std::move(entries_[i]);
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            const int64_t unchanged = reuse && gone.live.ids.data() == donor_.ids ? donor_.unchanged : 0;
            forget(gone);
            if (unchanged > 0) {
                reuse->captured_tokens = (int64_t) gone.live.ids.size();
                reuse->unchanged_tokens = unchanged;
                reuse->kv = std::move(gone.kv);
            }
            if (spill_) spilled_.push_back(std::move(gone));
            ++evictions_;
            return true;
        }
        return false;
    }
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0;
    size_t slot_bytes_ = 0, slot_count_ = 0;
    std::deque<SavedConversation> entries_; // least recently active first
    bool spill_ = false;
    std::vector<SavedConversation> spilled_;
    ConversationKvReuse reuse_;
    std::optional<SavedConversation> ready_;   // complete recovery metadata; its KV lives in reuse_
    uint64_t ready_dispatch_ = 0;
    const void* reuse_owner_ = nullptr;
    struct EvictionDonor {
        const int32_t* ids = nullptr;
        const void* owner = nullptr;
        int64_t unchanged = 0;
    } donor_;
};

} // namespace basalt::core
