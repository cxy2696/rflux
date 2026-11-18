#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::regimelab {

/*
 * Match target HMM states to reference states by minimising
 *
 *   sum_features [ (mean_t - mean_r)^2 / variance_r
 *                  + variance_weight * (log variance_t - log variance_r)^2 ]
 *
 * over all permutations.  `means` and `variances` are row-major
 * n_states x n_features panels.  Returns `order`, where reference state k
 * corresponds to target state order[k].  `variance_weight` must be finite and
 * non-negative.  The assignment is solved with the O(n^3)
 * shortest-augmenting-path Hungarian algorithm.
 */
struct AlignedHmmParameters {
    std::vector<double> start_prob;
    std::vector<double> transition;
    std::vector<double> means;
    std::vector<double> variances;
    std::vector<std::int32_t> order;
    double cost = 0.0;
};

/*
 * Match target states to reference states and return the target parameters
 * reordered to the reference labels.  `transition` is row-major K x K;
 * `means`/`variances` are row-major K x D.  The cost is the sum of the
 * matched assignment costs.
 */
[[nodiscard]] AlignedHmmParameters align_parameters(
    const double* reference_means, const double* reference_variances,
    const double* target_start_prob, const double* target_transition,
    const double* target_means, const double* target_variances,
    std::size_t n_states, std::size_t n_features, double variance_weight);

[[nodiscard]] std::vector<std::int32_t> match_states(
    const double* reference_means, const double* reference_variances,
    const double* target_means, const double* target_variances,
    std::size_t n_states, std::size_t n_features, double variance_weight);

}  // namespace rflux::regimelab
