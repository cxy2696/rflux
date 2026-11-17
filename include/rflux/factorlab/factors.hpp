#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "rflux/factorlab/factor_base.hpp"

namespace rflux::factorlab {

/*
 * All factors return a row-major n_obs x n_assets panel.  A cell is NaN during
 * warm-up and whenever an input needed by that cell is non-finite.
 */

/* close[t] / close[t-window] - 1. */
class MomentumFactor final : public PanelFactor {
public:
    explicit MomentumFactor(std::size_t window);

    [[nodiscard]] std::string_view name() const noexcept override {
        return "momentum";
    }
    [[nodiscard]] std::vector<double> compute(
        const core::FactorContext& context) const override;
};

/*
 * -trailing_return(window) * (volume[t] / mean(volume[t-window+1..t])).
 * The negative sign makes the factor a reversal signal; the volume ratio
 * amplifies moves on unusually active bars.
 */
class VolumeReversalFactor final : public PanelFactor {
public:
    explicit VolumeReversalFactor(std::size_t window);

    [[nodiscard]] std::string_view name() const noexcept override {
        return "volume_reversal";
    }
    [[nodiscard]] std::vector<double> compute(
        const core::FactorContext& context) const override;
};

/* (close[t] - rolling VWAP) / rolling VWAP, VWAP uses close * volume. */
class VwapDeviationFactor final : public PanelFactor {
public:
    explicit VwapDeviationFactor(std::size_t window);

    [[nodiscard]] std::string_view name() const noexcept override {
        return "vwap_deviation";
    }
    [[nodiscard]] std::vector<double> compute(
        const core::FactorContext& context) const override;
};

/*
 * Rolling Amihud illiquidity:
 *   scale * mean(|simple_return| / dollar_volume).
 * A larger value means a larger price move per unit of traded value.
 */
class AmihudFactor final : public PanelFactor {
public:
    explicit AmihudFactor(std::size_t window, double scale = 1e6);

    [[nodiscard]] std::string_view name() const noexcept override {
        return "amihud";
    }
    [[nodiscard]] std::vector<double> compute(
        const core::FactorContext& context) const override;

private:
    double scale_;
};

}  // namespace rflux::factorlab
