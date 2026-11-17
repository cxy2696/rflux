#pragma once

#include <cstddef>

#include "rflux/core/algorithm.hpp"
#include "rflux/core/registry.hpp"
#include "rflux/regimelab/hmm.hpp"

namespace rflux::regimelab {

struct RegimeSpec {
    std::size_t n_states = 2;
    std::size_t n_features = 1;
    HmmOptions options;
};

using RegimeDetectorFactory =
    core::AlgorithmRegistry<core::BaseRegimeDetector, const RegimeSpec&>;

void register_builtin_regime_detectors();

}  // namespace rflux::regimelab
