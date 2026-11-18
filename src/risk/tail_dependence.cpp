#include "rflux/risk/tail_dependence.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "internal/common.hpp"

namespace rflux::risk {

std::vector<double> tail_dependence(const double* returns, std::size_t n_obs,
                                    std::size_t n_assets, double q,
                                    bool upper) {
    rflux::detail::require_pointer(returns, "returns must not be NULL");
    rflux::detail::require(n_obs > 0, "n_obs must be positive");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    rflux::detail::require(std::isfinite(q) && q > 0.0 && q < 1.0,
                           "q must be in (0, 1)");
    rflux::detail::checked_mul(n_obs, n_assets, "n_obs*n_assets overflows");
    rflux::detail::checked_mul(n_assets, n_assets,
                               "n_assets*n_assets overflows");

    const double cutoff_q = upper ? 1.0 - q : q;
    std::vector<std::uint8_t> events(n_assets * n_obs, 0);
    std::vector<double> counts(n_assets, 0.0);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        std::vector<double> valid;
        valid.reserve(n_obs);
        for (std::size_t row = 0; row < n_obs; ++row) {
            const double value = returns[row * n_assets + asset];
            if (std::isfinite(value)) valid.push_back(value);
        }
        if (valid.empty()) continue;
        const double cutoff = rflux::detail::quantile_select(
            valid.data(), valid.size(), cutoff_q);
        for (std::size_t row = 0; row < n_obs; ++row) {
            const double value = returns[row * n_assets + asset];
            if (!std::isfinite(value)) continue;
            const bool selected = upper ? value >= cutoff : value <= cutoff;
            events[asset * n_obs + row] = selected ? 1 : 0;
            if (selected) counts[asset] += 1.0;
        }
    }

    std::vector<double> result(n_assets * n_assets,
                               rflux::detail::kNaN);
    for (std::size_t i = 0; i < n_assets; ++i) result[i * n_assets + i] = 1.0;
    for (std::size_t i = 0; i < n_assets; ++i) {
        for (std::size_t j = i + 1; j < n_assets; ++j) {
            if (counts[i] == 0.0 || counts[j] == 0.0) continue;
            std::size_t joint = 0;
            for (std::size_t row = 0; row < n_obs; ++row) {
                if (events[i * n_obs + row] && events[j * n_obs + row]) {
                    ++joint;
                }
            }
            const double value =
                0.5 * (static_cast<double>(joint) / counts[i] +
                       static_cast<double>(joint) / counts[j]);
            result[i * n_assets + j] = value;
            result[j * n_assets + i] = value;
        }
    }
    return result;
}

}  // namespace rflux::risk
