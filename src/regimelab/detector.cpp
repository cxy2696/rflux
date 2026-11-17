#include "rflux/regimelab/detector.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "internal/common.hpp"

namespace rflux::regimelab {

HmmRegimeDetector::HmmRegimeDetector(std::size_t n_states,
                                     std::size_t n_features,
                                     HmmOptions options)
    : model_(n_states, n_features), options_(options) {}

void HmmRegimeDetector::fit(const double* observations, std::size_t n_obs,
                            std::size_t n_features) {
    rflux::detail::require(n_features == model_.n_features(),
            "n_features does not match the detector");
    model_.fit(observations, n_obs, options_);
}

std::vector<std::int32_t> HmmRegimeDetector::states(
    const double* observations, std::size_t n_obs) const {
    return model_.viterbi(observations, n_obs);
}

std::vector<double> HmmRegimeDetector::probabilities(
    const double* observations, std::size_t n_obs) const {
    return model_.filtered(observations, n_obs);
}

}  // namespace rflux::regimelab
