#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace basalt::models::qwen4exp::serve {

// An incumbent and one nearby speculative row budget, compared in A/B/B/A windows.
template <std::size_t MaxSlots> class DecodeRowBudget {
    static constexpr int block = 16;
    struct Sample {
        double elapsed = 0.0;
        std::array<double, 2 * block> gaps{};
        std::array<int, MaxSlots> tokens{};
        int windows = 0;
        int rows = 0;

        double rate() const {
            int total = 0;
            for (int count : tokens) total += count;
            return elapsed > 0 ? 1000.0 * total / elapsed : 0.0;
        }
        double max_gap() const {
            return *std::max_element(gaps.begin(), gaps.begin() + windows);
        }
        double p95_gap() const {
            auto ordered = gaps;
            std::sort(ordered.begin(), ordered.begin() + windows);
            return ordered[(95 * windows + 99) / 100 - 1];
        }
    };

    struct Profile {
        uint64_t key = 0;
        uint64_t epoch = 0;
        int rows = 0;
        int demand = 0;
    };
    std::array<Profile, 16> profiles_{};
    uint64_t epoch_ = 0, trial_key_ = 0;
    uint64_t cohort_ = 0;
    int slots_ = 0, hard_ = 0, incumbent_ = 0, trial_ = 0;
    int cooldown_ = 0, step_ = 0;
    int qualified_demand_ = 0, trial_demand_ = 0;
    bool try_up_ = false;
    bool checked_profile_ = false, trial_from_profile_ = false;
    Sample base_, probe_;
    double first_base_ms_ = 0.0, first_base_rate_ = 0.0;
    std::array<double, MaxSlots> last_emit_{};

    Profile* profile(uint64_t key) {
        auto it = std::find_if(profiles_.begin(), profiles_.end(), [key](const Profile& p) { return p.key == key; });
        return it == profiles_.end() ? nullptr : &*it;
    }

    void clear_trial() {
        trial_ = step_ = trial_demand_ = 0;
        trial_key_ = 0;
        trial_from_profile_ = false;
        base_ = {};
        probe_ = {};
        first_base_ms_ = first_base_rate_ = 0.0;
    }

public:
    struct Decision {
        int prior = 0, candidate = 0;
        double base_rate = 0, probe_rate = 0;
        bool promoted = false, late = false;
        uint64_t profile_key = 0;
        bool remembered = false;
    };

    void reset() { slots_ = 0; }
    int trial() const { return slots_ ? trial_ : 0; }
    bool trial_from_profile() const { return trial() != 0 && trial_from_profile_; }
    bool starting_trial() const { return trial() != 0 && step_ == 0; }

    int choose(uint64_t cohort, int slots, int hard, int requested, uint64_t workload = 0) {
        if (cohort_ != cohort || slots_ != slots || hard_ != hard) {
            cohort_ = cohort;
            slots_ = slots;
            hard_ = incumbent_ = hard;
            qualified_demand_ = hard;
            cooldown_ = 24;
            try_up_ = checked_profile_ = false;
            clear_trial();
            last_emit_.fill(0);
            ++epoch_;
        }
        const int demand = std::min(hard_, requested);
        if (demand > qualified_demand_) {
            incumbent_ = hard_;
            qualified_demand_ = hard_;
            clear_trial();
            cooldown_ = 24;
        } else if (trial_ && demand > trial_demand_) {
            clear_trial();
            cooldown_ = 24;
        }
        if (trial_ == 0 && cooldown_ == 0 && demand > slots_ && demand >= incumbent_) {
            if (!checked_profile_ && workload) {
                if (const auto* remembered = profile(workload);
                    remembered && epoch_ - remembered->epoch <= 32 &&
                    remembered->rows >= slots_ && remembered->rows < hard_ && demand <= remembered->demand) {
                    const int adjacent = std::max(remembered->rows, incumbent_ - 1);
                    if (demand > adjacent) {
                        trial_ = adjacent;
                        trial_from_profile_ = true;
                    }
                }
            }
            checked_profile_ = true;
            if (!trial_) {
                const bool upward = incumbent_ < hard_ && (try_up_ || incumbent_ == slots_);
                const int adjacent = incumbent_ + (upward ? 1 : -1);
                if (adjacent >= slots_ && demand > std::min(incumbent_, adjacent))
                    trial_ = adjacent;
            }
            if (trial_) {
                trial_key_ = workload;
                trial_demand_ = demand;
            }
            try_up_ = !try_up_;
        }
        return trial_ && step_ >= block && step_ < 3 * block ? trial_ : incumbent_;
    }

    Decision observe(int rows, double elapsed_ms, const int* committed, const double* sent_ms) {
        double gap = 0;
        for (int i = 0; i < slots_; ++i) {
            if (last_emit_[i] > 0) gap = std::max(gap, sent_ms[i] - last_emit_[i]);
            last_emit_[i] = sent_ms[i];
        }
        if (trial_ == 0) {
            if (cooldown_ > 0) --cooldown_;
            return {};
        }
        const bool testing = step_ >= block && step_ < 3 * block;
        Sample& sample = testing ? probe_ : base_;
        sample.elapsed += elapsed_ms;
        sample.gaps[sample.windows++] = gap;
        sample.rows += rows;
        for (int i = 0; i < slots_; ++i) sample.tokens[i] += committed[i];
        const bool late = testing && gap > 1.2 * base_.max_gap();
        if (++step_ == block) {
            first_base_ms_ = base_.elapsed;
            first_base_rate_ = base_.rate();
        }
        if (step_ < 4 * block && !late) return {};

        Decision result{incumbent_, trial_, base_.rate(), probe_.rate(), false, late, trial_key_, trial_from_profile_};
        const double last_ms = base_.elapsed - first_base_ms_;
        const double last_rate = last_ms > 0.0
            ? (base_.rate() * base_.elapsed - first_base_rate_ * first_base_ms_) / last_ms : 0.0;
        const double stronger_rate = std::max(first_base_rate_, last_rate);
        if (!late && std::abs(probe_.rows - base_.rows) >= block / 2 &&
            std::min(first_base_rate_, last_rate) >= 0.90 * stronger_rate &&
            result.probe_rate > 1.03 * stronger_rate &&
            probe_.p95_gap() <= 1.1 * base_.p95_gap() && probe_.max_gap() <= 1.2 * base_.max_gap()) {
            result.promoted = true;
            for (int i = 0; i < slots_; ++i)
                result.promoted &= probe_.tokens[i] * base_.elapsed >= 0.95 * base_.tokens[i] * probe_.elapsed;
            if (result.promoted) {
                incumbent_ = trial_;
                qualified_demand_ = incumbent_ < hard_ ? trial_demand_ : hard_;
            }
        }
        if (trial_key_) {
            auto* remembered = profile(trial_key_);
            if (result.promoted && trial_ < hard_) {
                if (!remembered)
                    remembered = &*std::min_element(profiles_.begin(), profiles_.end(),
                        [](const Profile& a, const Profile& b) { return a.epoch < b.epoch; });
                *remembered = {trial_key_, epoch_, trial_, trial_demand_};
            } else if (remembered && (trial_from_profile_ || result.promoted)) {
                *remembered = {};
            }
        }
        cooldown_ = late ? 128 : 64;
        clear_trial();
        return result;
    }
};

}   // namespace basalt::models::qwen4exp::serve
