#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::regimelab {

/*
 * Diagonal-covariance Gaussian HMM.
 *
 * Inference conventions:
 * - filtered() is causal: P(s_t | x_0..x_t).
 * - smoothed() and viterbi() use the whole sample.  They are appropriate for
 *   retrospective attribution but must not be used as point-in-time signals.
 * - forecast() propagates a state distribution through the transition matrix;
 *   it does not forecast prices or returns.
 */

struct HmmOptions {
    std::size_t max_iter = 200;
    std::size_t n_init = 3;
    double tol = 1e-6;
    double variance_floor = 1e-6;
    std::uint64_t seed = 42;
    bool sort_states = true;
};

struct HmmParameters {
    std::vector<double> start_prob;   /* K */
    std::vector<double> transition;   /* K*K, row-major */
    std::vector<double> means;        /* K*D, row-major */
    std::vector<double> variances;    /* K*D, row-major */
};

class GaussianHMM {
public:
    GaussianHMM(std::size_t n_states, std::size_t n_features);

    /* Row-major n_obs x n_features finite observations. */
    void fit(const double* observations, std::size_t n_obs,
             const HmmOptions& options = {});

    void set_parameters(const HmmParameters& parameters);
    const HmmParameters& parameters() const;

    std::size_t n_states() const noexcept { return n_states_; }
    std::size_t n_features() const noexcept { return n_features_; }
    bool ready() const noexcept { return ready_; }
    bool has_fit_diagnostics() const noexcept { return has_fit_diagnostics_; }

    std::size_t iterations() const;
    bool converged() const;
    double log_likelihood() const;
    double aic() const;
    double bic() const;

    /*
     * Log-likelihood of observations under the current parameters.  Unlike
     * log_likelihood()/aic()/bic(), this works for loaded parameters and for
     * new data, so it can drive out-of-sample model selection.
     */
    double score(const double* observations, std::size_t n_obs) const;

    /* Causal filtered probabilities, row-major n_obs x K. */
    std::vector<double> filtered(const double* observations,
                                 std::size_t n_obs) const;

    /* Full-sample smoothed probabilities, row-major n_obs x K. */
    std::vector<double> smoothed(const double* observations,
                                 std::size_t n_obs) const;

    /* Full-sample Viterbi path. */
    std::vector<std::int32_t> viterbi(const double* observations,
                                      std::size_t n_obs) const;

    /* state_probs * transition^horizon. */
    std::vector<double> forecast(const double* state_probs,
                                 std::size_t n_states,
                                 std::size_t horizon) const;

    /* One causal update: posterior after observation given a prior. */
    std::vector<double> filter_step(const double* state_probs,
                                    const double* observation) const;

    /* Long-run distribution of the transition matrix. */
    std::vector<double> stationary_distribution() const;

private:
    void validate_observations(const double* observations,
                               std::size_t n_obs) const;
    void validate_model() const;

    std::size_t n_states_;
    std::size_t n_features_;
    HmmParameters parameters_;
    bool ready_ = false;
    bool has_fit_diagnostics_ = false;
    std::size_t iterations_ = 0;
    bool converged_ = false;
    double log_likelihood_ = 0.0;
    double aic_ = 0.0;
    double bic_ = 0.0;
};

}  // namespace rflux::regimelab
