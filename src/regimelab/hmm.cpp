#include "rflux/regimelab/hmm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "internal/common.hpp"
#include "internal/linalg.hpp"
#include "internal/parallel.hpp"

namespace rflux::regimelab {
namespace {

using rflux::detail::checked_mul;
using rflux::detail::kLog2Pi;
using rflux::detail::kNaN;
using rflux::detail::log_add_exp;
using rflux::detail::log_sum_exp;
using rflux::detail::require;
using rflux::detail::require_finite;
using rflux::detail::require_pointer;
using rflux::detail::Rng;
using rflux::detail::safe_log;

constexpr double kProbabilityFloor = 1e-12;
constexpr double kStateMassFloor = 1e-10;

struct EstepResult {
    std::vector<double> log_alpha;
    std::vector<double> log_beta;
    std::vector<double> log_emission;
    std::vector<double> gamma;
    double log_likelihood = 0.0;
};

struct ForwardResult {
    std::vector<double> log_alpha;
    double log_likelihood = 0.0;
};

/* Log-domain parameters computed once per EM iteration/inference call. */
struct LogModel {
    std::vector<double> start;
    std::vector<double> transition;
};

LogModel log_model(const HmmParameters& parameters, std::size_t k) {
    LogModel logs;
    logs.start.resize(k);
    logs.transition.resize(k * k);
    for (std::size_t state = 0; state < k; ++state) {
        logs.start[state] = safe_log(parameters.start_prob[state]);
    }
    for (std::size_t previous = 0; previous < k; ++previous) {
        for (std::size_t state = 0; state < k; ++state) {
            logs.transition[previous * k + state] = safe_log(
                parameters.transition[previous * k + state]);
        }
    }
    return logs;
}

struct Candidate {
    HmmParameters parameters;
    double log_likelihood = -std::numeric_limits<double>::infinity();
    std::size_t iterations = 0;
    bool converged = false;
};

void validate_options(const HmmOptions& options) {
    require(options.max_iter > 0, "max_iter must be positive");
    require(options.n_init > 0, "n_init must be positive");
    require(std::isfinite(options.tol) && options.tol > 0.0,
            "tol must be finite and positive");
    require(std::isfinite(options.variance_floor) &&
                options.variance_floor > 0.0,
            "variance_floor must be finite and positive");
}

std::vector<double> feature_means(const double* observations, std::size_t n,
                                  std::size_t d) {
    std::vector<long double> sums(d, 0.0L);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            sums[j] += static_cast<long double>(observations[i * d + j]);
        }
    }
    std::vector<double> means(d, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        means[j] = static_cast<double>(sums[j] /
                                       static_cast<long double>(n));
    }
    return means;
}

std::vector<double> feature_variances(const double* observations, std::size_t n,
                                      std::size_t d,
                                      const std::vector<double>& means) {
    std::vector<long double> sums(d, 0.0L);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            const long double diff =
                static_cast<long double>(observations[i * d + j]) -
                static_cast<long double>(means[j]);
            sums[j] += diff * diff;
        }
    }
    std::vector<double> variances(d, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        variances[j] = static_cast<double>(
            sums[j] / static_cast<long double>(n - 1));
    }
    return variances;
}

void validate_parameter_sizes(const HmmParameters& parameters, std::size_t k,
                              std::size_t d) {
    require(parameters.start_prob.size() == k,
            "start_prob must have length n_states");
    require(parameters.transition.size() == checked_mul(k, k, "K*K overflow"),
            "transition must have length n_states*n_states");
    const std::size_t state_features =
        checked_mul(k, d, "K*D overflow");
    require(parameters.means.size() == state_features,
            "means must have length n_states*n_features");
    require(parameters.variances.size() == state_features,
            "variances must have length n_states*n_features");
}

bool close_to_one(double value) {
    return std::abs(value - 1.0) <= 1e-8;
}

void normalize_with_floor(std::vector<double>& values) {
    long double sum = 0.0L;
    for (double value : values) sum += static_cast<long double>(value);
    if (!(sum > 0.0L)) {
        std::fill(values.begin(), values.end(),
                  1.0 / static_cast<double>(values.size()));
        return;
    }
    for (double& value : values) {
        value = static_cast<double>(static_cast<long double>(value) / sum);
        value = std::max(value, kProbabilityFloor);
    }
    sum = 0.0L;
    for (double value : values) sum += static_cast<long double>(value);
    for (double& value : values) {
        value = static_cast<double>(static_cast<long double>(value) / sum);
    }
}

/* Normalise a probability vector without changing structural zeros. */
void normalize_without_floor(std::vector<double>& values) {
    long double sum = 0.0L;
    for (double value : values) sum += static_cast<long double>(value);
    if (!(sum > 0.0L) || !std::isfinite(static_cast<double>(sum))) {
        throw NumericalError("HMM probability vector lost its mass");
    }
    for (double& value : values) {
        value = static_cast<double>(static_cast<long double>(value) / sum);
    }
}

void normalize_transition(std::vector<double>& transition, std::size_t k) {
    for (std::size_t row = 0; row < k; ++row) {
        long double sum = 0.0L;
        for (std::size_t col = 0; col < k; ++col) {
            sum += static_cast<long double>(transition[row * k + col]);
        }
        if (!(sum > 0.0L)) {
            for (std::size_t col = 0; col < k; ++col) {
                transition[row * k + col] = 1.0 / static_cast<double>(k);
            }
            continue;
        }
        for (std::size_t col = 0; col < k; ++col) {
            double& value = transition[row * k + col];
            value = static_cast<double>(static_cast<long double>(value) / sum);
            value = std::max(value, kProbabilityFloor);
        }
        sum = 0.0L;
        for (std::size_t col = 0; col < k; ++col) {
            sum += static_cast<long double>(transition[row * k + col]);
        }
        for (std::size_t col = 0; col < k; ++col) {
            transition[row * k + col] = static_cast<double>(
                static_cast<long double>(transition[row * k + col]) / sum);
        }
    }
}

void validate_probability_tables(const HmmParameters& parameters,
                                  std::size_t k) {
    long double start_sum = 0.0L;
    for (double value : parameters.start_prob) {
        require(std::isfinite(value) && value >= 0.0,
                "start probabilities must be finite and non-negative");
        start_sum += static_cast<long double>(value);
    }
    require(close_to_one(static_cast<double>(start_sum)),
            "start probabilities must sum to one");
    for (std::size_t row = 0; row < k; ++row) {
        long double row_sum = 0.0L;
        for (std::size_t col = 0; col < k; ++col) {
            const double value = parameters.transition[row * k + col];
            require(std::isfinite(value) && value >= 0.0,
                    "transition probabilities must be finite and non-negative");
            row_sum += static_cast<long double>(value);
        }
        require(close_to_one(static_cast<double>(row_sum)),
                "each transition row must sum to one");
    }
}

void validate_emission_parameters(const HmmParameters& parameters) {
    for (std::size_t i = 0; i < parameters.means.size(); ++i) {
        require(std::isfinite(parameters.means[i]), "means must be finite");
        require(std::isfinite(parameters.variances[i]) &&
                    parameters.variances[i] > 0.0,
                "variances must be finite and positive");
    }
}

long double squared_distance(const double* observations, std::size_t left,
                           std::size_t right, std::size_t d) {
    long double distance = 0.0L;
    for (std::size_t j = 0; j < d; ++j) {
        const long double diff =
            static_cast<long double>(observations[left * d + j]) -
            static_cast<long double>(observations[right * d + j]);
        distance += diff * diff;
    }
    return distance;
}

std::size_t nearest_center(const double* observations, std::size_t row,
                          const std::vector<std::size_t>& centers,
                          std::size_t k, std::size_t d) {
    long double best_distance =
        std::numeric_limits<long double>::infinity();
    std::size_t best = 0;
    for (std::size_t c = 0; c < k; ++c) {
        const long double distance =
            squared_distance(observations, row, centers[c], d);
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }
    return best;
}

void sort_states(HmmParameters& parameters, std::size_t k, std::size_t d) {
    std::vector<std::size_t> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         const double mean_a = parameters.means[a * d];
                         const double mean_b = parameters.means[b * d];
                         if (mean_a != mean_b) return mean_a < mean_b;
                         const double var_a = parameters.variances[a * d];
                         const double var_b = parameters.variances[b * d];
                         if (var_a != var_b) return var_a < var_b;
                         return a < b;
                     });

    HmmParameters sorted;
    sorted.start_prob.resize(k);
    sorted.transition.resize(k * k);
    sorted.means.resize(k * d);
    sorted.variances.resize(k * d);
    for (std::size_t a = 0; a < k; ++a) {
        sorted.start_prob[a] = parameters.start_prob[order[a]];
        for (std::size_t j = 0; j < d; ++j) {
            sorted.means[a * d + j] = parameters.means[order[a] * d + j];
            sorted.variances[a * d + j] =
                parameters.variances[order[a] * d + j];
        }
        for (std::size_t b = 0; b < k; ++b) {
            sorted.transition[a * k + b] =
                parameters.transition[order[a] * k + order[b]];
        }
    }
    parameters = std::move(sorted);
}

std::vector<double> log_emission_matrix(const HmmParameters& parameters,
                                        std::size_t k, std::size_t d,
                                        const double* observations,
                                        std::size_t n) {
    std::vector<double> log_emission(checked_mul(n, k, "n*K overflow"), 0.0);
    /*
     * The Gaussian normaliser and 1/variance depend only on the state and
     * feature, never on the observation index.  Hoisting them out of the
     * time loop removes a log and a division for every (t, state, feature)
     * cell while keeping the log-domain formulation unchanged.
     */
    std::vector<long double> normaliser(k, 0.0L);
    std::vector<long double> inverse_variance(
        checked_mul(k, d, "K*D overflow"), 0.0L);
    for (std::size_t state = 0; state < k; ++state) {
        long double value = 0.0L;
        for (std::size_t j = 0; j < d; ++j) {
            const long double variance = static_cast<long double>(
                parameters.variances[state * d + j]);
            value += -0.5L * static_cast<long double>(kLog2Pi) -
                     0.5L * std::log(variance);
            inverse_variance[state * d + j] = 1.0L / variance;
        }
        normaliser[state] = value;
    }

    for (std::size_t t = 0; t < n; ++t) {
        const double* row = observations + t * d;
        for (std::size_t state = 0; state < k; ++state) {
            long double value = normaliser[state];
            for (std::size_t j = 0; j < d; ++j) {
                const long double diff =
                    static_cast<long double>(row[j]) -
                    static_cast<long double>(parameters.means[state * d + j]);
                value -= 0.5L * diff * diff * inverse_variance[state * d + j];
            }
            log_emission[t * k + state] = static_cast<double>(value);
        }
    }
    return log_emission;
}

ForwardResult forward_pass(const LogModel& logs, std::size_t k,
                           const std::vector<double>& log_emission,
                           std::size_t n) {
    ForwardResult result;
    result.log_alpha.assign(
        checked_mul(n, k, "n*K overflow"),
        -std::numeric_limits<double>::infinity());
    for (std::size_t state = 0; state < k; ++state) {
        result.log_alpha[state] = logs.start[state] + log_emission[state];
    }
    std::vector<double> terms(k, 0.0);
    for (std::size_t t = 1; t < n; ++t) {
        for (std::size_t state = 0; state < k; ++state) {
            for (std::size_t previous = 0; previous < k; ++previous) {
                terms[previous] =
                    result.log_alpha[(t - 1) * k + previous] +
                    logs.transition[previous * k + state];
            }
            result.log_alpha[t * k + state] =
                log_sum_exp(terms.data(), k) +
                log_emission[t * k + state];
        }
    }
    result.log_likelihood =
        log_sum_exp(result.log_alpha.data() + (n - 1) * k, k);
    if (!std::isfinite(result.log_likelihood)) {
        throw NumericalError("HMM forward likelihood is not finite");
    }
    return result;
}

EstepResult expectation_step(const HmmParameters& parameters,
                             const LogModel& logs, std::size_t k,
                             std::size_t d, const double* observations,
                             std::size_t n) {
    EstepResult result;
    const std::size_t count = checked_mul(n, k, "n*K overflow");
    result.log_emission =
        log_emission_matrix(parameters, k, d, observations, n);
    ForwardResult forward =
        forward_pass(logs, k, result.log_emission, n);
    result.log_alpha = std::move(forward.log_alpha);
    result.log_likelihood = forward.log_likelihood;
    result.log_beta.assign(count, 0.0);
    result.gamma.assign(count, 0.0);

    std::vector<double> terms(k, 0.0);
    for (std::size_t state = 0; state < k; ++state) {
        result.log_beta[(n - 1) * k + state] = 0.0;
    }
    for (std::size_t t = n - 1; t-- > 0;) {
        for (std::size_t previous = 0; previous < k; ++previous) {
            for (std::size_t state = 0; state < k; ++state) {
                terms[state] =
                    logs.transition[previous * k + state] +
                    result.log_emission[(t + 1) * k + state] +
                    result.log_beta[(t + 1) * k + state];
            }
            result.log_beta[t * k + previous] =
                log_sum_exp(terms.data(), k);
        }
    }

    for (std::size_t t = 0; t < n; ++t) {
        long double row_sum = 0.0L;
        for (std::size_t state = 0; state < k; ++state) {
            const double log_weight =
                result.log_alpha[t * k + state] +
                result.log_beta[t * k + state] -
                result.log_likelihood;
            result.gamma[t * k + state] = std::exp(log_weight);
            row_sum += static_cast<long double>(result.gamma[t * k + state]);
        }
        if (!(row_sum > 0.0L) || !std::isfinite(static_cast<double>(row_sum))) {
            throw NumericalError("HMM posterior has zero or invalid mass");
        }
        for (std::size_t state = 0; state < k; ++state) {
            result.gamma[t * k + state] = static_cast<double>(
                static_cast<long double>(result.gamma[t * k + state]) /
                row_sum);
        }
    }
    return result;
}

HmmParameters maximization_step(const EstepResult& estep,
                                const LogModel& logs, std::size_t k,
                                std::size_t d, const double* observations,
                                std::size_t n,
                                const std::vector<double>& global_means,
                                const std::vector<double>& variance_floors) {
    HmmParameters updated;
    updated.start_prob.assign(k, 0.0);
    updated.transition.assign(k * k, 0.0);
    updated.means.assign(k * d, 0.0);
    updated.variances.assign(k * d, 0.0);

    for (std::size_t state = 0; state < k; ++state) {
        updated.start_prob[state] = std::max(
            estep.gamma[state], kProbabilityFloor);
    }
    normalize_with_floor(updated.start_prob);

    for (std::size_t t = 0; t + 1 < n; ++t) {
        for (std::size_t previous = 0; previous < k; ++previous) {
            for (std::size_t state = 0; state < k; ++state) {
                const double log_xi =
                    estep.log_alpha[t * k + previous] +
                    logs.transition[previous * k + state] +
                    estep.log_emission[(t + 1) * k + state] +
                    estep.log_beta[(t + 1) * k + state] -
                    estep.log_likelihood;
                updated.transition[previous * k + state] += std::exp(log_xi);
            }
        }
    }
    normalize_transition(updated.transition, k);

    std::vector<double> state_mass(k, 0.0);
    for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t state = 0; state < k; ++state) {
            state_mass[state] += estep.gamma[t * k + state];
        }
    }

    for (std::size_t state = 0; state < k; ++state) {
        if (!(state_mass[state] > kStateMassFloor)) {
            for (std::size_t j = 0; j < d; ++j) {
                updated.means[state * d + j] = global_means[j];
                updated.variances[state * d + j] = variance_floors[j];
            }
            continue;
        }
        const long double mass =
            static_cast<long double>(state_mass[state]);
        for (std::size_t j = 0; j < d; ++j) {
            long double sum = 0.0L;
            for (std::size_t t = 0; t < n; ++t) {
                sum += static_cast<long double>(estep.gamma[t * k + state]) *
                       static_cast<long double>(observations[t * d + j]);
            }
            updated.means[state * d + j] =
                static_cast<double>(sum / mass);
        }
        for (std::size_t j = 0; j < d; ++j) {
            long double sum = 0.0L;
            const double mean = updated.means[state * d + j];
            for (std::size_t t = 0; t < n; ++t) {
                const double diff = observations[t * d + j] - mean;
                sum += static_cast<long double>(estep.gamma[t * k + state]) *
                       static_cast<long double>(diff * diff);
            }
            const double variance = static_cast<double>(sum / mass);
            updated.variances[state * d + j] =
                std::max(variance_floors[j], variance);
        }
    }
    return updated;
}

HmmParameters kmeans_initial_parameters(
    const double* observations, std::size_t n, std::size_t d, std::size_t k,
    const std::vector<double>& global_means,
    const std::vector<double>& variance_floors, Rng& rng) {
    std::vector<std::size_t> centers(k, 0);
    centers[0] = rng.uniform_index(static_cast<std::uint32_t>(
        std::min<std::size_t>(n, std::numeric_limits<std::uint32_t>::max())));

    std::vector<long double> min_distance(
        n, std::numeric_limits<long double>::infinity());
    for (std::size_t c = 1; c < k; ++c) {
        std::size_t last_center = centers[c - 1];
        long double total = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            long double distance = 0.0L;
            for (std::size_t j = 0; j < d; ++j) {
                const long double diff =
                    static_cast<long double>(observations[i * d + j]) -
                    static_cast<long double>(
                        observations[last_center * d + j]);
                distance += diff * diff;
            }
            min_distance[i] = std::min(min_distance[i], distance);
            total += min_distance[i];
        }
        std::size_t selected = 0;
        if (!(total > 0.0L) || !std::isfinite(static_cast<double>(total))) {
            selected = static_cast<std::size_t>(rng.next_u64()) % n;
        } else {
            const long double target =
                static_cast<long double>(rng.uniform()) * total;
            long double cumulative = 0.0L;
            selected = n - 1;
            for (std::size_t i = 0; i < n; ++i) {
                cumulative += min_distance[i];
                if (cumulative >= target) {
                    selected = i;
                    break;
                }
            }
        }
        centers[c] = selected;
    }

    std::vector<std::size_t> labels(n, 0);
    std::vector<std::size_t> previous_labels(n, std::numeric_limits<std::size_t>::max());
    for (std::size_t iteration = 0; iteration < 50; ++iteration) {
        bool changed = false;
        for (std::size_t i = 0; i < n; ++i) {
            labels[i] = nearest_center(observations, i, centers, k, d);
            if (labels[i] != previous_labels[i]) changed = true;
        }
        if (!changed) break;
        previous_labels = labels;

        std::vector<std::size_t> counts(k, 0);
        for (std::size_t i = 0; i < n; ++i) {
            ++counts[labels[i]];
        }
        // This initialiser is k-medoids: each center remains an observed point.
        // The EM step below replaces these centers with emission means.
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            std::size_t nearest = centers[c];
            long double nearest_distance =
                std::numeric_limits<long double>::infinity();
            for (std::size_t i = 0; i < n; ++i) {
                if (labels[i] != c) continue;
                const long double distance =
                    squared_distance(observations, i, centers[c], d);
                if (distance < nearest_distance) {
                    nearest_distance = distance;
                    nearest = i;
                }
            }
            centers[c] = nearest;
        }
    }

    // Final hard assignments and state moments.
    std::vector<std::size_t> counts(k, 0);
    std::vector<long double> sums(k * d, 0.0L);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t best =
            nearest_center(observations, i, centers, k, d);
        labels[i] = best;
        ++counts[best];
        for (std::size_t j = 0; j < d; ++j) {
            sums[best * d + j] +=
                static_cast<long double>(observations[i * d + j]);
        }
    }

    HmmParameters parameters;
    parameters.start_prob.assign(k, kProbabilityFloor);
    parameters.transition.assign(k * k, kProbabilityFloor);
    parameters.means.assign(k * d, 0.0);
    parameters.variances.assign(k * d, 0.0);

    for (std::size_t state = 0; state < k; ++state) {
        if (counts[state] == 0) {
            for (std::size_t j = 0; j < d; ++j) {
                parameters.means[state * d + j] = global_means[j];
                parameters.variances[state * d + j] = variance_floors[j];
            }
            continue;
        }
        for (std::size_t j = 0; j < d; ++j) {
            parameters.means[state * d + j] = static_cast<double>(
                sums[state * d + j] /
                static_cast<long double>(counts[state]));
        }
        for (std::size_t j = 0; j < d; ++j) {
            long double sum = 0.0L;
            const double mean = parameters.means[state * d + j];
            for (std::size_t i = 0; i < n; ++i) {
                if (labels[i] != state) continue;
                const long double diff =
                    static_cast<long double>(observations[i * d + j]) -
                    static_cast<long double>(mean);
                sum += diff * diff;
            }
            const double variance = counts[state] > 1
                                        ? static_cast<double>(
                                              sum / static_cast<long double>(
                                                        counts[state] - 1))
                                        : variance_floors[j];
            parameters.variances[state * d + j] =
                std::max(variance_floors[j], variance);
        }
    }

    parameters.start_prob[labels.front()] += 1.0;
    normalize_with_floor(parameters.start_prob);
    for (std::size_t t = 0; t + 1 < n; ++t) {
        parameters.transition[labels[t] * k + labels[t + 1]] += 1.0;
    }
    normalize_transition(parameters.transition, k);
    return parameters;
}

Candidate train_once(const double* observations, std::size_t n, std::size_t d,
                     std::size_t k, const HmmOptions& options,
                     const std::vector<double>& global_means,
                     const std::vector<double>& variance_floors,
                     std::uint64_t seed) {
    Rng rng(seed);
    HmmParameters candidate = kmeans_initial_parameters(
        observations, n, d, k, global_means, variance_floors, rng);

    Candidate best;
    double previous_likelihood = -std::numeric_limits<double>::infinity();
    const std::size_t max_iter = options.max_iter;
    for (std::size_t iteration = 1; iteration <= max_iter; ++iteration) {
        const LogModel logs = log_model(candidate, k);
        const EstepResult estep =
            expectation_step(candidate, logs, k, d, observations, n);
        if (estep.log_likelihood > best.log_likelihood) {
            best.parameters = candidate;
            best.log_likelihood = estep.log_likelihood;
        }
        if (iteration > 1) {
            const double improvement =
                estep.log_likelihood - previous_likelihood;
            const double tolerance =
                options.tol * (1.0 + std::abs(previous_likelihood));
            if (improvement <= tolerance && improvement >= -tolerance) {
                best.iterations = iteration;
                best.converged = true;
                break;
            }
            if (improvement < -tolerance) {
                // A large decrease can only come from numerical/floor effects.
                // Keep the best parameters seen so far and stop this restart.
                best.iterations = iteration;
                break;
            }
        }
        previous_likelihood = estep.log_likelihood;
        best.iterations = iteration;
        if (iteration == max_iter) break;
        candidate = maximization_step(estep, logs, k, d, observations, n,
                                      global_means, variance_floors);
    }
    return best;
}

}  // namespace

GaussianHMM::GaussianHMM(std::size_t n_states, std::size_t n_features) {
    require(n_states > 0, "n_states must be positive");
    require(n_features > 0, "n_features must be positive");
    require(n_states <= static_cast<std::size_t>(
                            std::numeric_limits<std::int32_t>::max()),
            "n_states must fit in int32_t");
    checked_mul(n_states, n_features, "n_states*n_features overflows size_t");
    checked_mul(n_states, n_states, "n_states*n_states overflows size_t");
    n_states_ = n_states;
    n_features_ = n_features;

    parameters_.start_prob.assign(n_states, 1.0 / static_cast<double>(n_states));
    parameters_.transition.assign(n_states * n_states,
                                  1.0 / static_cast<double>(n_states));
    parameters_.means.assign(n_states * n_features, 0.0);
    parameters_.variances.assign(n_states * n_features, 1.0);
    ready_ = false;
}

void GaussianHMM::validate_observations(const double* observations,
                                        std::size_t n_obs) const {
    require_pointer(observations, "observations must not be NULL");
    require(n_obs > 0, "n_obs must be positive");
    require_finite(observations,
                   checked_mul(n_obs, n_features_,
                               "n_obs*n_features overflows size_t"),
                   "observations");
}

void GaussianHMM::validate_model() const {
    if (!ready_) {
        throw NotReady("HMM is not fitted or parameterised; call fit or set_parameters first");
    }
    validate_parameter_sizes(parameters_, n_states_, n_features_);
    validate_probability_tables(parameters_, n_states_);
    validate_emission_parameters(parameters_);
}

void GaussianHMM::validate_inference(const double* observations,
                                     std::size_t n_obs) const {
    validate_model();
    validate_observations(observations, n_obs);
}

void GaussianHMM::require_fit_diagnostics() const {
    if (!has_fit_diagnostics_) throw NotReady("HMM has no fit diagnostics");
}

void GaussianHMM::fit(const double* observations, std::size_t n_obs,
                      const HmmOptions& options) {
    validate_options(options);
    validate_observations(observations, n_obs);
    require(n_obs >= 2, "HMM fitting requires at least two observations");
    require(n_obs >= n_states_,
            "HMM fitting requires n_obs >= n_states");

    const std::vector<double> global_means =
        feature_means(observations, n_obs, n_features_);
    const std::vector<double> global_variances =
        feature_variances(observations, n_obs, n_features_, global_means);
    std::vector<double> variance_floors(n_features_);
    for (std::size_t j = 0; j < n_features_; ++j) {
        variance_floors[j] = std::max(
            1e-12, options.variance_floor *
                       std::max(global_variances[j], 1e-12));
    }

    /*
     * Restarts are independent: each draws its own seed from (seed, init) and
     * never reads another restart's state.  Running them in parallel therefore
     * preserves the result exactly; the winner is still selected in restart
     * order with a strict comparison.  The worker cap bounds peak memory,
     * because every restart holds O(n_obs * n_states) ln-space buffers.
     */
    constexpr std::size_t kMaxParallelRestarts = 4;
    const std::size_t restart_threads =
        (n_obs >= 512 && options.n_init >= 2) ? kMaxParallelRestarts : 1;
    std::vector<Candidate> candidates(options.n_init);
    detail::parallel_for(
        options.n_init,
        [&](std::size_t init) {
            const std::uint64_t seed = detail::mix_seed(
                options.seed, static_cast<std::uint64_t>(init));
            try {
                candidates[init] = train_once(
                    observations, n_obs, n_features_, n_states_, options,
                    global_means, variance_floors, seed);
            } catch (const NumericalError&) {
                candidates[init] = Candidate{};
            }
        },
        restart_threads);

    Candidate overall;
    for (std::size_t init = 0; init < options.n_init; ++init) {
        if (candidates[init].log_likelihood > overall.log_likelihood) {
            overall = std::move(candidates[init]);
        }
    }
    if (!std::isfinite(overall.log_likelihood)) {
        throw NumericalError("HMM fitting failed for every restart");
    }

    HmmParameters fitted = overall.parameters;
    if (options.sort_states) {
        sort_states(fitted, n_states_, n_features_);
    }
    const double n = static_cast<double>(n_obs);
    const double free_parameters =
        static_cast<double>(n_states_ - 1) +
        static_cast<double>(n_states_ * (n_states_ - 1)) +
        2.0 * static_cast<double>(n_states_ * n_features_);

    parameters_ = std::move(fitted);
    ready_ = true;
    has_fit_diagnostics_ = true;
    iterations_ = overall.iterations;
    converged_ = overall.converged;
    log_likelihood_ = overall.log_likelihood;
    aic_ = -2.0 * log_likelihood_ + 2.0 * free_parameters;
    bic_ = -2.0 * log_likelihood_ + free_parameters * std::log(n);
}

void GaussianHMM::set_parameters(const HmmParameters& parameters) {
    validate_parameter_sizes(parameters, n_states_, n_features_);
    validate_probability_tables(parameters, n_states_);
    validate_emission_parameters(parameters);
    parameters_ = parameters;
    ready_ = true;
    has_fit_diagnostics_ = false;
    iterations_ = 0;
    converged_ = false;
    log_likelihood_ = kNaN;
    aic_ = kNaN;
    bic_ = kNaN;
}

const HmmParameters& GaussianHMM::parameters() const {
    validate_model();
    return parameters_;
}

std::size_t GaussianHMM::iterations() const {
    require_fit_diagnostics();
    return iterations_;
}

bool GaussianHMM::converged() const {
    require_fit_diagnostics();
    return converged_;
}

double GaussianHMM::log_likelihood() const {
    require_fit_diagnostics();
    return log_likelihood_;
}

double GaussianHMM::aic() const {
    require_fit_diagnostics();
    return aic_;
}

double GaussianHMM::bic() const {
    require_fit_diagnostics();
    return bic_;
}

double GaussianHMM::score(const double* observations,
                          std::size_t n_obs) const {
    validate_inference(observations, n_obs);
    const std::vector<double> log_emission = log_emission_matrix(
        parameters_, n_states_, n_features_, observations, n_obs);
    const LogModel logs = log_model(parameters_, n_states_);
    return forward_pass(logs, n_states_, log_emission, n_obs).log_likelihood;
}

std::vector<double> GaussianHMM::filtered(const double* observations,
                                          std::size_t n_obs) const {
    validate_inference(observations, n_obs);
    const std::size_t count =
        checked_mul(n_obs, n_states_, "n_obs*K overflows size_t");
    const std::vector<double> log_emission = log_emission_matrix(
        parameters_, n_states_, n_features_, observations, n_obs);
    const LogModel logs = log_model(parameters_, n_states_);
    ForwardResult forward =
        forward_pass(logs, n_states_, log_emission, n_obs);
    const std::vector<double>& log_alpha = forward.log_alpha;

    std::vector<double> probabilities(count, kNaN);
    for (std::size_t t = 0; t < n_obs; ++t) {
        const double normalizer =
            log_sum_exp(log_alpha.data() + t * n_states_, n_states_);
        if (!std::isfinite(normalizer)) {
            throw NumericalError("HMM filtering produced invalid mass");
        }
        long double sum = 0.0L;
        for (std::size_t state = 0; state < n_states_; ++state) {
            probabilities[t * n_states_ + state] =
                std::exp(log_alpha[t * n_states_ + state] - normalizer);
            sum += static_cast<long double>(
                probabilities[t * n_states_ + state]);
        }
        for (std::size_t state = 0; state < n_states_; ++state) {
            probabilities[t * n_states_ + state] = static_cast<double>(
                static_cast<long double>(
                    probabilities[t * n_states_ + state]) /
                sum);
        }
    }
    return probabilities;
}

std::vector<double> GaussianHMM::smoothed(const double* observations,
                                          std::size_t n_obs) const {
    validate_inference(observations, n_obs);
    const LogModel logs = log_model(parameters_, n_states_);
    const EstepResult result = expectation_step(
        parameters_, logs, n_states_, n_features_, observations, n_obs);
    return result.gamma;
}

std::vector<std::int32_t> GaussianHMM::viterbi(const double* observations,
                                               std::size_t n_obs) const {
    validate_inference(observations, n_obs);
    const std::vector<double> log_emission = log_emission_matrix(
        parameters_, n_states_, n_features_, observations, n_obs);
    const LogModel logs = log_model(parameters_, n_states_);
    const std::size_t count =
        checked_mul(n_obs, n_states_, "n_obs*K overflows size_t");
    std::vector<double> score(count,
                              -std::numeric_limits<double>::infinity());
    std::vector<std::int32_t> backpointer(count, 0);

    for (std::size_t state = 0; state < n_states_; ++state) {
        score[state] = logs.start[state] + log_emission[state];
    }
    for (std::size_t t = 1; t < n_obs; ++t) {
        for (std::size_t state = 0; state < n_states_; ++state) {
            double best = -std::numeric_limits<double>::infinity();
            std::size_t best_previous = 0;
            for (std::size_t previous = 0; previous < n_states_; ++previous) {
                const double candidate =
                    score[(t - 1) * n_states_ + previous] +
                    logs.transition[previous * n_states_ + state];
                if (candidate > best) {
                    best = candidate;
                    best_previous = previous;
                }
            }
            score[t * n_states_ + state] =
                best + log_emission[t * n_states_ + state];
            backpointer[t * n_states_ + state] =
                static_cast<std::int32_t>(best_previous);
        }
    }

    std::size_t best_last = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t state = 0; state < n_states_; ++state) {
        if (score[(n_obs - 1) * n_states_ + state] > best_score) {
            best_score = score[(n_obs - 1) * n_states_ + state];
            best_last = state;
        }
    }
    if (!std::isfinite(best_score)) {
        throw NumericalError("Viterbi path has no finite score");
    }

    std::vector<std::int32_t> path(n_obs, 0);
    std::size_t state = best_last;
    path[n_obs - 1] = static_cast<std::int32_t>(state);
    for (std::size_t t = n_obs - 1; t-- > 0;) {
        state = static_cast<std::size_t>(
            backpointer[(t + 1) * n_states_ + state]);
        path[t] = static_cast<std::int32_t>(state);
    }
    return path;
}

std::vector<double> GaussianHMM::forecast(const double* state_probs,
                                          std::size_t n_states,
                                          std::size_t horizon) const {
    validate_model();
    require_pointer(state_probs, "state_probs must not be NULL");
    require(n_states == n_states_, "state_probs length must equal n_states");
    long double total = 0.0L;
    for (std::size_t i = 0; i < n_states_; ++i) {
        require(std::isfinite(state_probs[i]) && state_probs[i] >= 0.0,
                "state_probs must be finite and non-negative");
        total += static_cast<long double>(state_probs[i]);
    }
    require(total > 0.0L, "state_probs must have positive mass");

    std::vector<double> result(n_states_, 0.0);
    for (std::size_t i = 0; i < n_states_; ++i) {
        result[i] = static_cast<double>(
            static_cast<long double>(state_probs[i]) / total);
    }
    if (horizon == 0) return result;

    std::vector<double> power = parameters_.transition;
    std::vector<double> next(n_states_, 0.0);
    while (horizon > 0) {
        if ((horizon & 1U) != 0U) {
            std::fill(next.begin(), next.end(), 0.0);
            for (std::size_t from = 0; from < n_states_; ++from) {
                for (std::size_t to = 0; to < n_states_; ++to) {
                    next[to] += result[from] * power[from * n_states_ + to];
                }
            }
            result.swap(next);
            normalize_without_floor(result);
        }
        horizon >>= 1U;
        if (horizon == 0) break;
        std::vector<double> square(n_states_ * n_states_, 0.0);
        for (std::size_t i = 0; i < n_states_; ++i) {
            for (std::size_t j = 0; j < n_states_; ++j) {
                long double sum = 0.0L;
                for (std::size_t mid = 0; mid < n_states_; ++mid) {
                    sum += static_cast<long double>(
                               power[i * n_states_ + mid]) *
                           static_cast<long double>(
                               power[mid * n_states_ + j]);
                }
                square[i * n_states_ + j] = static_cast<double>(sum);
            }
        }
        power.swap(square);
    }
    return result;
}

std::vector<double> GaussianHMM::filter_step(const double* state_probs,
                                             const double* observation) const {
    validate_model();
    require_pointer(state_probs, "state_probs must not be NULL");
    require_pointer(observation, "observation must not be NULL");
    require_finite(observation, n_features_, "observation");
    long double prior_total = 0.0L;
    for (std::size_t state = 0; state < n_states_; ++state) {
        require(std::isfinite(state_probs[state]) && state_probs[state] >= 0.0,
                "state_probs must be finite and non-negative");
        prior_total += static_cast<long double>(state_probs[state]);
    }
    require(prior_total > 0.0L, "state_probs must have positive mass");

    const LogModel logs = log_model(parameters_, n_states_);
    const std::vector<double> emission = log_emission_matrix(
        parameters_, n_states_, n_features_, observation, 1);
    std::vector<double> terms(n_states_, 0.0);
    std::vector<double> log_post(n_states_, 0.0);
    for (std::size_t state = 0; state < n_states_; ++state) {
        for (std::size_t previous = 0; previous < n_states_; ++previous) {
            terms[previous] = safe_log(state_probs[previous]) +
                              logs.transition[previous * n_states_ + state];
        }
        log_post[state] =
            log_sum_exp(terms.data(), n_states_) + emission[state];
    }
    const double normalizer =
        log_sum_exp(log_post.data(), n_states_);
    if (!std::isfinite(normalizer)) {
        throw NumericalError("HMM filter step has no finite posterior mass");
    }
    std::vector<double> out(n_states_, 0.0);
    for (std::size_t state = 0; state < n_states_; ++state) {
        out[state] = std::exp(log_post[state] - normalizer);
    }
    return out;
}

std::vector<double> GaussianHMM::stationary_distribution() const {
    validate_model();
    const std::size_t k = n_states_;
    if (k == 1) return {1.0};

    // Solve (A^T - I) pi = 0 with the last equation replaced by sum(pi)=1.
    std::vector<long double> matrix(k * k, 0.0L);
    std::vector<long double> rhs(k, 0.0L);
    for (std::size_t row = 0; row < k; ++row) {
        for (std::size_t col = 0; col < k; ++col) {
            matrix[row * k + col] =
                static_cast<long double>(
                    parameters_.transition[col * k + row]);
        }
        matrix[row * k + row] -= 1.0L;
    }
    for (std::size_t col = 0; col < k; ++col) {
        matrix[(k - 1) * k + col] = 1.0L;
    }
    rhs[k - 1] = 1.0L;

    const std::vector<long double> solution =
        detail::solve_dense(std::move(matrix), std::move(rhs), k,
                            "HMM stationary distribution");
    long double total = 0.0L;
    for (long double value : solution) {
        if (value < -1e-12L) {
            throw NumericalError("stationary distribution has negative mass");
        }
        total += std::max(value, 0.0L);
    }
    if (!(total > 0.0L)) {
        throw NumericalError("stationary distribution lost probability mass");
    }
    std::vector<double> out(k, 0.0);
    for (std::size_t state = 0; state < k; ++state) {
        out[state] = static_cast<double>(
            std::max(solution[state], 0.0L) / total);
    }
    return out;
}

std::vector<double> GaussianHMM::expected_durations() const {
    validate_model();
    std::vector<double> out(n_states_, 0.0);
    for (std::size_t state = 0; state < n_states_; ++state) {
        const double stay = parameters_.transition[state * n_states_ + state];
        out[state] = stay >= 1.0
                         ? std::numeric_limits<double>::infinity()
                         : 1.0 / (1.0 - stay);
    }
    return out;
}

}  // namespace rflux::regimelab
