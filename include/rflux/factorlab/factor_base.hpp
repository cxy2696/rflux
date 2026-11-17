#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rflux/core/algorithm.hpp"

namespace rflux::factorlab {

/*
 * Shared panel plumbing for time-series factors.
 *
 * Derived classes implement one cell function; this base validates the panel
 * shape, allocates the output and applies the asset/time traversal.  Cell
 * functions return NaN for warm-up or missing input.
 */
class PanelFactor : public core::BaseFactor {
public:
    [[nodiscard]] std::size_t window() const noexcept { return window_; }

protected:
    explicit PanelFactor(std::size_t window, bool needs_volume);

    void validate_context(const core::FactorContext& context) const;

    template <typename Cell>
    [[nodiscard]] std::vector<double> compute_cells(
        const core::FactorContext& context, Cell&& cell) const {
        validate_context(context);
        const std::size_t n_assets = context.close.n_assets;
        std::vector<double> output(
            context.close.n_obs * n_assets,
            std::numeric_limits<double>::quiet_NaN());
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            for (std::size_t t = 0; t < context.close.n_obs; ++t) {
                output[t * n_assets + asset] =
                    static_cast<double>(cell(t, asset));
            }
        }
        return output;
    }

private:
    std::size_t window_;
    bool needs_volume_;
};

}  // namespace rflux::factorlab
