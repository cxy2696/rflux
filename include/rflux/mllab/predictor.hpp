#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "rflux/core/algorithm.hpp"
#include "rflux/core/market.hpp"

namespace rflux::mllab {

/*
 * Regime-aware forecast contract.
 *
 * The lower-level core::BasePredictor is a pure feature -> target regressor used
 * by Huber and future tabular models.  BaseForecastModel adds the portfolio-facing
 * contract: confidence must have the same length as expected_returns, and the
 * model may condition on an integer regime label.
 */
class BaseForecastModel : public core::BaseAlgorithm {
public:
    virtual void fit(const core::MatrixView& features,
                     const std::vector<double>& target) = 0;
    [[nodiscard]] virtual core::PredictionResult predict(
        const core::MatrixView& features, int regime) const = 0;
    [[nodiscard]] virtual bool fitted() const noexcept = 0;
};

}  // namespace rflux::mllab
