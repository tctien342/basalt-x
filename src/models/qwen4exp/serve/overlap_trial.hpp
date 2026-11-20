#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace basalt::models::qwen4exp::serve {

// One replay of an independently qualified candidate. Changing the cohort
// invalidates the comparison, while a slower valid cycle rejects it.
template <std::size_t Slots>
class OverlapTrial {
public:
    using Tokens = std::array<int64_t, Slots>;
    enum class State { opening, probing, confirming, promoted, rejected, interrupted };

    OverlapTrial(int64_t layout, int64_t remembered, std::size_t decoders, double gap_budget_ms = 0)
        : incumbent_(std::min<int64_t>(1024, layout)),
          candidate_(remembered),
          decoders_(decoders),
          gap_budget_ms_(gap_budget_ms),
          state_(remembered > 0 && remembered <= layout && remembered % 256 == 0 &&
                         remembered != incumbent_ && decoders > 0 && decoders <= Slots &&
                         std::isfinite(gap_budget_ms) && gap_budget_ms >= 0
                     ? State::opening : State::interrupted) {}

    int64_t chunk() const {
        return state_ == State::probing || state_ == State::promoted ? candidate_ : incumbent_;
    }

    State state() const { return state_; }
    bool settled_winner() const { return state_ == State::promoted && healthy_blocks_ >= 2; }
    double opening_gap_ms() const { return opening_.worst_gap_ms; }
    double confirmed_gap_ms() const { return closing_.worst_gap_ms; }
    double gap_limit_ms() const {
        if (gap_budget_ms_ > 0) return gap_budget_ms_;
        // Both incumbent blocks are fixed before promotion. Candidate timing never updates the gap budget.
        const double reference = state_ == State::promoted || closing_.cycles > 0
            ? std::max(opening_.worst_gap_ms, closing_.worst_gap_ms) : opening_.worst_gap_ms;
        return 1.20 * reference;
    }

    const char* interruption_reason() const { return interruption_reason_; }

    void interrupt(const char* reason = "cohort_changed") {
        interruption_reason_ = reason;
        state_ = State::interrupted;
    }

    // A partial cycle before settlement carries latency evidence only.
    void check_terminal_gap(double gap_ms) {
        if (state_ != State::promoted) return;
        if (!std::isfinite(gap_ms) || gap_ms <= 0) {
            interrupt("invalid_terminal_gap");
        } else if (gap_ms > gap_limit_ms()) {
            interruption_reason_ = "terminal_gap";
            state_ = State::rejected;
        }
    }

    void observe(int64_t actual, double cycle_ms, double worst_gap_ms, const Tokens& emitted) {
        if (state_ == State::rejected || state_ == State::interrupted) return;
        if (actual != chunk() || !std::isfinite(cycle_ms) || cycle_ms <= 0 ||
            !std::isfinite(worst_gap_ms) || worst_gap_ms <= 0 ||
            std::any_of(emitted.begin(), emitted.begin() + decoders_, [](int64_t count) { return count < 0; })) {
            interrupt("invalid_sample");
            return;
        }
        if (warmup_left_ > 0) {
            --warmup_left_;
            return;
        }
        const bool using_candidate = state_ == State::probing || state_ == State::promoted;
        if ((using_candidate || gap_budget_ms_ > 0) && worst_gap_ms > gap_limit_ms()) {
            if (using_candidate) state_ = State::rejected;
            else interrupt("incumbent_gap_budget");
            return;
        }
        auto& block = state_ == State::opening ? opening_ : state_ == State::confirming ? closing_ : probing_;
        block.ms += cycle_ms;
        block.prefill_tokens += actual;
        block.worst_gap_ms = std::max(worst_gap_ms, block.worst_gap_ms);
        for (std::size_t i = 0; i < decoders_; ++i) block.emitted[i] += emitted[i];
        if (++block.cycles != 2) return;
        if (state_ == State::opening) state_ = State::probing;
        else if (state_ == State::probing) state_ = State::confirming;
        else if (state_ == State::confirming) {
            if (const char* reason = baseline_drift(opening_, closing_)) interrupt(reason);
            else state_ = beats(opening_, probing_) && beats(closing_, probing_) ? State::promoted : State::rejected;
            probing_ = {};
        } else {
            const bool holds = beats(opening_, probing_) && beats(closing_, probing_);
            healthy_blocks_ = holds ? std::min(2, healthy_blocks_ + 1) : 0;
            slow_blocks_ = holds ? 0 : slow_blocks_ + 1;
            if (slow_blocks_ == 2) state_ = State::rejected;
            probing_ = {};
        }
    }

private:
    struct Block {
        double ms = 0;
        int64_t prefill_tokens = 0;
        double worst_gap_ms = 0;
        Tokens emitted{};
        int cycles = 0;
    };

    const char* baseline_drift(const Block& a, const Block& b) const {
        const auto compatible = [&](int64_t pa, int64_t pb) {
            const double x = pa * b.ms, y = pb * a.ms;
            return std::min(x, y) > 0 && std::min(x, y) >= 0.90 * std::max(x, y);
        };
        if (!compatible(a.prefill_tokens, b.prefill_tokens)) return "prefill_rate_drift";
        for (std::size_t i = 0; i < decoders_; ++i)
            if (!compatible(a.emitted[i], b.emitted[i])) return "slot_rate_drift";
        return nullptr;
    }

    bool beats(const Block& incumbent, const Block& candidate) const {
        if (candidate.prefill_tokens * incumbent.ms < 1.03 * incumbent.prefill_tokens * candidate.ms)
            return false;
        for (std::size_t i = 0; i < decoders_; ++i)
            if (incumbent.emitted[i] <= 0 || candidate.emitted[i] <= 0 ||
                candidate.emitted[i] * incumbent.ms < 0.95 * incumbent.emitted[i] * candidate.ms)
                return false;
        return true;
    }

    const int64_t incumbent_, candidate_;
    const std::size_t decoders_;
    const double gap_budget_ms_;
    State state_;
    const char* interruption_reason_ = "none";
    int warmup_left_ = 2;
    int slow_blocks_ = 0, healthy_blocks_ = 0;
    Block opening_, probing_, closing_;
};

}  // namespace basalt::models::qwen4exp::serve
