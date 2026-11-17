#include "rflux/factorlab/factors.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "internal/common.hpp"

namespace rflux::factorlab {
namespace {

using rflux::detail::kNaN;
using rflux::detail::require;

double cell(const core::PanelView& panel, std::size_t t, std::size_t asset) {
    return panel.data[t * panel.n_assets + asset];
}

double trailing_return(const core::PanelView& close, std::size_t t,
                       std::size_t asset, std::size_t window) {
    if (t < window) return kNaN;
    const double current = cell(close, t, asset);
    const double base = cell(close, t - window, asset);
    if (!std::isfinite(current) || !std::isfinite(base) || base == 0.0) {
        return kNaN;
    }
    return current / base - 1.0;
}

}  // namespace

PanelFactor::PanelFactor(std::size_t window, bool needs_volume)
    : window_(window), needs_volume_(needs_volume) {
    require(window_ > 0, "factor window must be positive");
}

void PanelFactor::validate_context(const core::FactorContext& context) const {
    core::detail::require_panel(context.close, "close panel");
    if (needs_volume_) {
        core::detail::require_panel(context.volume, "volume panel");
        core::detail::require_same_shape(context.close, context.volume,
                                         "close and volume panels must match");
    }
}

MomentumFactor::MomentumFactor(std::size_t window)
    : PanelFactor(window, false) {}

std::vector<double> MomentumFactor::compute(
    const core::FactorContext& context) const {
    return compute_cells(context, [&](std::size_t t, std::size_t asset) {
        return trailing_return(context.close, t, asset, window());
    });
}

VolumeReversalFactor::VolumeReversalFactor(std::size_t window)
    : PanelFactor(window, true) {}

std::vector<double> VolumeReversalFactor::compute(
    const core::FactorContext& context) const {
    return compute_cells(context, [&](std::size_t t, std::size_t asset) {
        const double return_value =
            trailing_return(context.close, t, asset, window());
        if (!std::isfinite(return_value)) return kNaN;
        long double volume_sum = 0.0L;
        for (std::size_t i = t - window() + 1; i <= t; ++i) {
            const double volume = cell(context.volume, i, asset);
            if (!std::isfinite(volume) || volume < 0.0) return kNaN;
            volume_sum += static_cast<long double>(volume);
        }
        if (!(volume_sum > 0.0L)) return kNaN;
        const double mean_volume =
            static_cast<double>(volume_sum / static_cast<long double>(window()));
        if (!(mean_volume > 0.0)) return kNaN;
        const double volume_ratio =
            cell(context.volume, t, asset) / mean_volume;
        if (!std::isfinite(volume_ratio)) return kNaN;
        return -return_value * volume_ratio;
    });
}

VwapDeviationFactor::VwapDeviationFactor(std::size_t window)
    : PanelFactor(window, true) {}

std::vector<double> VwapDeviationFactor::compute(
    const core::FactorContext& context) const {
    return compute_cells(context, [&](std::size_t t, std::size_t asset) {
        if (t + 1 < window()) return kNaN;
        long double weighted = 0.0L;
        long double volume_sum = 0.0L;
        for (std::size_t i = t + 1 - window(); i <= t; ++i) {
            const double price = cell(context.close, i, asset);
            const double volume = cell(context.volume, i, asset);
            if (!std::isfinite(price) || !std::isfinite(volume) ||
                volume < 0.0) {
                return kNaN;
            }
            weighted += static_cast<long double>(price) *
                        static_cast<long double>(volume);
            volume_sum += static_cast<long double>(volume);
        }
        if (!(volume_sum > 0.0L)) return kNaN;
        const double vwap = static_cast<double>(weighted / volume_sum);
        const double current = cell(context.close, t, asset);
        if (!(vwap > 0.0) || !std::isfinite(current)) return kNaN;
        return current / vwap - 1.0;
    });
}

AmihudFactor::AmihudFactor(std::size_t window, double scale)
    : PanelFactor(window, true), scale_(scale) {
    require(std::isfinite(scale_) && scale_ > 0.0,
            "Amihud scale must be finite and positive");
}

std::vector<double> AmihudFactor::compute(
    const core::FactorContext& context) const {
    return compute_cells(context, [&](std::size_t t, std::size_t asset) {
        if (t < window()) return kNaN;
        long double sum = 0.0L;
        for (std::size_t i = t - window() + 1; i <= t; ++i) {
            const double price = cell(context.close, i, asset);
            const double previous = cell(context.close, i - 1, asset);
            const double volume = cell(context.volume, i, asset);
            if (!std::isfinite(price) || !std::isfinite(previous) ||
                !std::isfinite(volume) || previous == 0.0 || price <= 0.0 ||
                volume <= 0.0) {
                return kNaN;
            }
            const double return_value = std::abs(price / previous - 1.0);
            const long double dollar_volume =
                static_cast<long double>(price) *
                static_cast<long double>(volume);
            if (!(dollar_volume > 0.0L)) return kNaN;
            sum += static_cast<long double>(return_value) / dollar_volume;
        }
        return static_cast<double>(sum /
                                   static_cast<long double>(window())) *
               scale_;
    });
}

}  // namespace rflux::factorlab
