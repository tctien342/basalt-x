#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace basalt::core {

// Snapshot-owned host storage. The ordinary copy owns independent bytes.
// An explicit parked-prefix share keeps the payload until either owner writes it.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    static constexpr size_t io_alignment = 4096;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer& other) : size_(other.size_) {
        segments_.reserve(other.segments_.size());
        for (const auto& segment : other.segments_) segments_.push_back(segment.clone());
    }
    ConversationBuffer& operator=(const ConversationBuffer& other) {
        if (this != &other) *this = ConversationBuffer(other);
        return *this;
    }
    ConversationBuffer shared_copy() const {
        ConversationBuffer out;
        out.segments_ = segments_;
        out.size_ = size_;
        return out;
    }
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    // The RAM cache charges every image its full logical capacity. Physical
    // admission subtracts only storage this capture can overwrite in place.
    size_t bytes(bool only_unique = false) const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_)
            if (!only_unique || !segment.shared()) n += segment.capacity() + Segment::overhead;
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0))) return SIZE_MAX;
        const size_t added = full + (tail != 0);
        if (added > SIZE_MAX / Segment::overhead || !add(total, added * Segment::overhead)) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, added)) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) { grow_or_shrink(n, &value); }
    /// The same without writing the new bytes: for a caller that overwrites every one of them (a read from a file).
    void resize_for_overwrite(size_t n) { grow_or_shrink(n, nullptr); }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    // a segment's bytes are written by the caller (a fill value or a copy), never zeroed first by the vector itself
    template <class T>
    struct NoInit : std::allocator<T> {
        template <class U> struct rebind { using other = NoInit<U>; };
        NoInit() = default;
        template <class U> NoInit(const NoInit<U>&) noexcept {}
        T* allocate(size_t n) {
            if (n < io_alignment || n > SIZE_MAX / sizeof(T)) return std::allocator<T>::allocate(n);
            return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{io_alignment}));
        }
        void deallocate(T* p, size_t n) noexcept {
            if (n < io_alignment) std::allocator<T>::deallocate(p, n);
            else ::operator delete(p, std::align_val_t{io_alignment});
        }
        template <class U> void construct(U* p) { ::new (static_cast<void*>(p)) U; }
        template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
    };
    using SegmentStorage = std::vector<uint8_t, NoInit<uint8_t>>;
    class Segment {
    public:
        static constexpr size_t overhead = sizeof(SegmentStorage) + 2 * sizeof(void*);
        size_t size() const { return storage_->size(); }
        size_t capacity() const { return storage_->capacity(); }
        bool shared() const { return storage_.use_count() > 1; }
        const uint8_t* data() const { return storage_->data(); }
        uint8_t* data() { return write().data(); }
        void reserve(size_t n) { write().reserve(n); }
        void resize(size_t n) { write().resize(n); }
        void resize(size_t n, uint8_t value) { write().resize(n, value); }
        Segment clone() const {
            Segment out;
            *out.storage_ = *storage_;
            return out;
        }

    private:
        SegmentStorage& write() {
            if (shared()) {
                auto out = std::make_shared<SegmentStorage>();
                out->reserve(capacity());
                out->resize(size());
                if (size()) std::memcpy(out->data(), storage_->data(), size());
                storage_ = std::move(out);
            }
            return *storage_;
        }
        std::shared_ptr<SegmentStorage> storage_ = std::make_shared<SegmentStorage>();
    };
    void grow_or_shrink(size_t n, const uint8_t* value) {
        if (n > size_) {
            const bool growing = size_ != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                if (value) segments_.back().resize(segments_.back().size() + extend, *value);
                else segments_.back().resize(segments_.back().size() + extend);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing));
                if (value) segment.resize(count, *value);
                else segment.resize(count);
                segments_.push_back(std::move(segment));
                size_ += count;
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    static size_t segment_capacity(size_t count, size_t previous, bool growing) {
        if (!growing) return count;
        return std::max(count, std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

} // namespace basalt::core
