#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace basalt::models::qwen4exp::serve {

class InterleavedPrefillChunk {
public:
    explicit InterleavedPrefillChunk(int64_t layout)
        : limit_(layout < kStep ? layout : layout / kStep * kStep),
          tokens_(std::min<int64_t>(1024, limit_)) {}

    int64_t tokens() const { return tokens_; }
    double reference_gap_ms() const { return reference_gap_ms_; }

    double decode_budget(double prefill_ms) const {
        return prefill_ms * decode_share_;
    }

    void observe(double gap_ms, double prefill_ms, bool one_change = false,
                 int64_t decoded = 0, double cycle_ms = 0.0) {
        if (gap_ms <= 0.0 || prefill_ms <= 0.0) return;
        if (samples_ < baseline_.size()) {
            baseline_[samples_++] = gap_ms;
            if (one_change && cycle_ms > 0.0) {
                reference_tokens_ += decoded;
                reference_ms_ += cycle_ms;
            }
            if (samples_ == baseline_.size()) {
                std::sort(baseline_.begin(), baseline_.end());
                reference_gap_ms_ = baseline_[1];
            }
            return;
        }
        double share = std::clamp((reference_gap_ms_ - prefill_ms) / prefill_ms, one_change ? 0.5 : 0.25, 0.75);
        if (one_change && reference_tokens_ > 0 && cycle_ms > 0.0 &&
            decoded * reference_ms_ < 0.95 * reference_tokens_ * cycle_ms)
            share = std::max(share, 0.625);
        const auto tune_share = [&] { decode_share_ += (share - decode_share_) * 0.25; };
        if (!one_change) tune_share();
        if (gap_ms > reference_gap_ms_ * 1.10) {
            tokens_ = std::max(std::min(kStep, limit_), tokens_ - kStep);
            healthy_ = 0;
            return;
        }
        const int64_t growth = std::min(kStep, limit_ - tokens_);
        const auto predicted_gap = gap_ms + prefill_ms * growth / tokens_;
        if (growth <= 0 || predicted_gap > reference_gap_ms_ * 0.90) {
            healthy_ = 0;
            if (one_change) tune_share();
            return;
        }
        if (++healthy_ == 4) {
            tokens_ += growth;
            healthy_ = 0;
        } else if (one_change) {
            tune_share();
        }
    }

private:
    static constexpr int64_t kStep = 256;
    int64_t limit_, tokens_;
    std::array<double, 3> baseline_{};
    std::size_t samples_ = 0;
    double reference_gap_ms_ = 0.0;
    double decode_share_ = 0.5;
    double reference_ms_ = 0.0;
    int64_t reference_tokens_ = 0;
    int healthy_ = 0;
};

class ShortPrefillTiming {
public:
    bool use_windows(int64_t tokens) const {
        return windows_.samples < 4 || batched_.samples < 4 ||
            windows_.average * tokens <= batched_.average * 1.2;
    }

    void observe(int64_t tokens, double ms, bool windows) {
        if (tokens <= 0 || ms <= 0.0) return;
        auto& sample = windows ? windows_ : batched_;
        const double cost = windows ? ms / tokens : ms;
        if (sample.samples == 0) sample.average = cost;
        else sample.average += (cost - sample.average) * 0.25;
        sample.samples = std::min(4, sample.samples + 1);
    }

private:
    struct Timing { double average = 0.0; int samples = 0; };
    Timing windows_, batched_;
};

}  // namespace basalt::models::qwen4exp::serve
