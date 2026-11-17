#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "rflux/core/algorithm.hpp"
#include "rflux/regimelab/hmm.hpp"

namespace rflux::regimelab {

/* BaseRegimeDetector adapter over the existing diagonal Gaussian HMM. */
class HmmRegimeDetector final : public core::BaseRegimeDetector {
public:
    explicit HmmRegimeDetector(std::size_t n_states,
                               std::size_t n_features = 1,
                               HmmOptions options = {});

    [[nodiscard]] std::string_view name() const noexcept override {
        return "gaussian_hmm";
    }
    void fit(const double* observations, std::size_t n_obs,
             std::size_t n_features) override;
    /* Full-sample Viterbi path; use probabilities() for causal filtering. */
    [[nodiscard]] std::vector<std::int32_t> states(
        const double* observations, std::size_t n_obs) const override;
    /* Causal filtered probabilities, row-major n_obs x n_states. */
    [[nodiscard]] std::vector<double> probabilities(
        const double* observations, std::size_t n_obs) const override;

    [[nodiscard]] GaussianHMM& model() noexcept { return model_; }
    [[nodiscard]] const GaussianHMM& model() const noexcept { return model_; }

private:
    GaussianHMM model_;
    HmmOptions options_;
};

}  // namespace rflux::regimelab
