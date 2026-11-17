#pragma once

#include <cstddef>

#include "rflux/core/algorithm.hpp"
#include "rflux/core/registry.hpp"

namespace rflux::mllab {

struct PredictorSpec {
    double epsilon = 1.35;
    std::size_t max_iter = 100;
    double tol = 1e-8;
    bool fit_intercept = true;
};

using PredictorFactory =
    core::AlgorithmRegistry<core::BasePredictor, const PredictorSpec&>;

void register_builtin_predictors();

}  // namespace rflux::mllab
