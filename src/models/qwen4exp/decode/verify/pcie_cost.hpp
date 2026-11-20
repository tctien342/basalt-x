#pragma once

#include <algorithm>
#include <cmath>

namespace basalt::core {

struct PcieLayerCost {
    double cpu_per_miss = 0.0, local = 0.0, peer = 0.0;
    double copy_start = 35.0, bytes_per_us = 0.0, gpu_per_expert = 0.0;
    double error_us = 0.0;
    bool peer_censored = false;
    bool measured_stage = false;   // only a completed transfer/compute can justify raising the soft cap
};

enum class PcieBudget { no_timing, peer_unobserved, cpu_not_last, hard_cap, no_savings, stage, count };

// All durations are microseconds from the router's doorbell. Fit whole experts in the remaining byte
// and latency budget without moving work off a CPU branch that is already ahead of the other tiers.
inline int pcie_layer_experts(const PcieLayerCost& c, int misses, int slots, double bytes, double hard_cap,
                              PcieBudget* reason = nullptr) {
    const auto decision = [reason](int n, PcieBudget why) {
        if (reason != nullptr) *reason = why;
        return n;
    };
    const double cpu = c.cpu_per_miss * misses;
    const double margin = std::max(15.0, c.error_us * misses);
    const double floor = std::max(c.local, c.peer);
    if (misses <= 0 || bytes <= 0.0 || c.cpu_per_miss <= 0.0 ||
        c.bytes_per_us <= 0.0 || c.gpu_per_expert <= 0.0) return decision(0, PcieBudget::no_timing);
    if (c.peer_censored) return decision(0, PcieBudget::peer_unobserved);
    if (cpu <= floor + margin) return decision(0, PcieBudget::cpu_not_last);

    const int max_count = std::min(slots, (int) (misses * hard_cap));
    if (max_count <= 0) return decision(0, PcieBudget::hard_cap);
    const int nominal = std::min(max_count, misses / 5);
    auto finish = [&](int n, double bytes_per_us, double gpu_per_expert) {
        const double staged = n * gpu_per_expert;
        const double gpu = std::max(c.local, c.copy_start + n * bytes / bytes_per_us) + staged;
        return std::max({c.cpu_per_miss * (misses - n), c.peer, gpu});
    };
    int best = 0;
    double best_us = cpu;
    for (int n = 1; n <= nominal; ++n) {
        const double us = finish(n, c.bytes_per_us, c.gpu_per_expert);
        if (us + margin < best_us) { best_us = us; best = n; }
    }
    const double low_bandwidth = c.bytes_per_us * 0.90, high_compute = c.gpu_per_expert * 1.10;
    // A useful whole expert can be larger than the soft 20% share on a sparse layer.
    // Bootstrap only one, with twice the margin and pessimistic transfer/compute costs.
    if (!c.measured_stage && nominal == 0 && misses > 1 &&
        finish(1, low_bandwidth, high_compute) + 2 * margin < cpu)
        return decision(1, PcieBudget::stage);
    if (c.measured_stage) {
        const double target = best_us - margin;
        for (int n = nominal + 1; n <= max_count; ++n) {
            const double copy_slack = target - c.copy_start - n * high_compute;
            if (copy_slack <= 0.0 || n * bytes > copy_slack * low_bandwidth) break;
            const double us = finish(n, low_bandwidth, high_compute);
            if (us + margin < best_us) { best_us = us; best = n; }
        }
    }
    return decision(best, best > 0 ? PcieBudget::stage : PcieBudget::no_savings);
}

}  // namespace basalt::core
