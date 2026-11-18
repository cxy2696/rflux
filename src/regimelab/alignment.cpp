#include "rflux/regimelab/alignment.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "internal/common.hpp"

namespace rflux::regimelab {
namespace {

using rflux::detail::kNaN;

void require_parameter_panel(const double* values, std::size_t n_states,
                             std::size_t n_features, const char* name) {
    rflux::detail::require_pointer(values, name);
    rflux::detail::require(n_states > 0, "n_states must be positive");
    rflux::detail::require(n_features > 0, "n_features must be positive");
    rflux::detail::checked_mul(n_states, n_features,
                               "state parameter dimensions overflow");
    for (std::size_t i = 0; i < n_states * n_features; ++i) {
        if (!std::isfinite(values[i])) {
            throw rflux::NonFinite(std::string(name) +
                                   " must contain finite values");
        }
    }
}

/* Minimum-cost assignment on a square cost matrix, computed with the
 * shortest augmenting path method and dual potentials.  Ties resolve in the
 * same order as the reference implementation, so state ids stay stable for
 * repeated fits of the same data. */
std::vector<std::size_t> hungarian(const std::vector<double>& cost,
                                   std::size_t n) {
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<std::size_t> p(n + 1, 0);
    std::vector<std::size_t> way(n + 1, 0);
    std::vector<double> minv(n + 1, 0.0);
    std::vector<bool> used(n + 1, false);

    for (std::size_t row = 1; row <= n; ++row) {
        p[0] = row;
        std::size_t column = 0;
        std::fill(minv.begin(), minv.end(), infinity);
        std::fill(used.begin(), used.end(), false);
        do {
            used[column] = true;
            const std::size_t current = p[column];
            double delta = infinity;
            std::size_t next = 0;
            for (std::size_t candidate = 1; candidate <= n; ++candidate) {
                if (used[candidate]) continue;
                const double reduced =
                    cost[(current - 1) * n + (candidate - 1)] -
                    u[current] - v[candidate];
                if (reduced < minv[candidate]) {
                    minv[candidate] = reduced;
                    way[candidate] = column;
                }
                if (minv[candidate] < delta) {
                    delta = minv[candidate];
                    next = candidate;
                }
            }
            for (std::size_t candidate = 0; candidate <= n; ++candidate) {
                if (used[candidate]) {
                    u[p[candidate]] += delta;
                    v[candidate] -= delta;
                } else {
                    minv[candidate] -= delta;
                }
            }
            column = next;
        } while (p[column] != 0);
        do {
            const std::size_t previous = way[column];
            p[column] = p[previous];
            column = previous;
        } while (column != 0);
    }

    std::vector<std::size_t> assignment(n, 0);
    for (std::size_t column = 1; column <= n; ++column) {
        if (p[column] == 0 || p[column] > n) {
            throw rflux::NumericalError("state matching produced no assignment");
        }
        assignment[p[column] - 1] = column - 1;
    }
    return assignment;
}

}  // namespace

std::vector<std::int32_t> match_states(const double* reference_means,
                                       const double* reference_variances,
                                       const double* target_means,
                                       const double* target_variances,
                                       std::size_t n_states,
                                       std::size_t n_features,
                                       double variance_weight) {
    require_parameter_panel(reference_means, n_states, n_features,
                            "reference_means must not be NULL");
    require_parameter_panel(reference_variances, n_states, n_features,
                            "reference_variances must not be NULL");
    require_parameter_panel(target_means, n_states, n_features,
                            "target_means must not be NULL");
    require_parameter_panel(target_variances, n_states, n_features,
                            "target_variances must not be NULL");
    rflux::detail::require(std::isfinite(variance_weight) &&
                               variance_weight >= 0.0,
                           "variance_weight must be finite and non-negative");
    for (std::size_t i = 0; i < n_states * n_features; ++i) {
        if (!(reference_variances[i] > 0.0) ||
            !(target_variances[i] > 0.0)) {
            throw rflux::InvalidArgument("variances must be positive");
        }
    }

    rflux::detail::checked_mul(n_states, n_states,
                               "state cost matrix dimensions overflow");
    std::vector<double> cost(n_states * n_states, kNaN);
    for (std::size_t reference = 0; reference < n_states; ++reference) {
        for (std::size_t target = 0; target < n_states; ++target) {
            double mean_cost = 0.0;
            double variance_cost = 0.0;
            for (std::size_t feature = 0; feature < n_features; ++feature) {
                const std::size_t offset = reference * n_features + feature;
                const std::size_t target_offset =
                    target * n_features + feature;
                const double scale =
                    std::sqrt(reference_variances[offset]);
                const double difference =
                    target_means[target_offset] - reference_means[offset];
                mean_cost +=
                    difference * difference / (scale * scale);
                const double log_ratio =
                    std::log(target_variances[target_offset]) -
                    std::log(reference_variances[offset]);
                variance_cost += log_ratio * log_ratio;
            }
            cost[reference * n_states + target] =
                mean_cost + variance_weight * variance_cost;
        }
    }

    for (double value : cost) {
        if (!std::isfinite(value)) {
            throw rflux::InvalidArgument(
                "state alignment cost must be finite");
        }
    }

    const std::vector<std::size_t> assignment =
        hungarian(cost, n_states);
    std::vector<std::int32_t> order(n_states, 0);
    for (std::size_t reference = 0; reference < n_states; ++reference) {
        order[reference] = static_cast<std::int32_t>(assignment[reference]);
    }
    return order;
}

}  // namespace rflux::regimelab
