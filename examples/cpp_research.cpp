// C++ core example: factors -> HMM regimes -> correlation network -> diffusion.
// Build: make examples && ./build/example_cpp_research
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "rflux/rflux.hpp"

int main() {
    constexpr std::size_t days = 520;
    constexpr std::size_t assets = 5;
    std::vector<double> panel(days * assets, 0.0);
    std::vector<double> market(days, 0.0);

    // Two volatility/mean regimes with a common market factor.  This is
    // synthetic data, not a claim about any real market.
    for (std::size_t t = 0; t < days; ++t) {
        const bool stress = t > days * 2 / 3;
        const double drift = stress ? -0.0006 : 0.0007;
        const double shock = stress ? 0.022 : 0.008;
        const double common =
            drift + shock * std::sin(static_cast<double>(t) * 0.31);
        market[t] = common;
        for (std::size_t a = 0; a < assets; ++a) {
            const double idio =
                0.004 * std::sin(static_cast<double>(t) * (0.17 + 0.03 * a) +
                                 static_cast<double>(a));
            panel[t * assets + a] =
                0.6 * common + idio + 0.0002 * static_cast<double>(a);
        }
    }

    std::printf("=== factors ===\n");
    for (std::size_t a = 0; a < assets; ++a) {
        std::vector<double> series(days);
        for (std::size_t t = 0; t < days; ++t) {
            series[t] = panel[t * assets + a];
        }
        const auto prices = [&] {
            std::vector<double> result(days, 1.0);
            for (std::size_t t = 1; t < days; ++t) {
                result[t] = result[t - 1] * (1.0 + series[t]);
            }
            return result;
        }();
        const auto momentum =
            rflux::qflux::rolling_mean(prices.data(), prices.size(), 60);
        const auto slope =
            rflux::qflux::rolling_slope(prices.data(), prices.size(), 60);
        const auto r2 =
            rflux::qflux::rolling_r2(prices.data(), prices.size(), 60);
        std::printf(
            "asset %zu: price=%.3f mom60=%.3f slope60=%.5f r2=%.3f\n", a,
            prices.back(), momentum.back(), slope.back(), r2.back());
    }

    std::printf("\n=== HMM regimes (market return) ===\n");
    rflux::regimelab::GaussianHMM hmm(3, 1);
    rflux::regimelab::HmmOptions options;
    options.max_iter = 200;
    options.n_init = 4;
    options.seed = 2024;
    hmm.fit(market.data(), market.size(), options);
    const auto probabilities = hmm.filtered(market.data(), market.size());
    std::vector<std::size_t> counts(3, 0);
    for (std::size_t t = 0; t < days; ++t) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < 3; ++k) {
            if (probabilities[t * 3 + k] >
                probabilities[t * 3 + best]) {
                best = k;
            }
        }
        ++counts[best];
    }
    std::printf(
        "states: %zu/%zu/%zu | loglik=%.2f | BIC=%.2f | converged=%s\n",
        counts[0], counts[1], counts[2], hmm.log_likelihood(), hmm.bic(),
        hmm.converged() ? "yes" : "no");

    const auto regimes = rflux::regimelab::volatility_regime(
        market.data(), market.size(), 20, 120, 0.25, 0.75);
    std::size_t high_vol = 0;
    for (std::int32_t label : regimes.labels) high_vol += label == 2;
    std::printf("causal high-volatility days: %zu\n", high_vol);

    std::printf("\n=== correlation network (last 120 days) ===\n");
    std::vector<double> window(120 * assets);
    std::copy(panel.end() - static_cast<std::ptrdiff_t>(120 * assets),
              panel.end(), window.begin());
    const auto corr = rflux::qflux::correlation_matrix(
        window.data(), 120, assets);
    std::vector<double> adjacency(assets * assets, 0.0);
    for (std::size_t i = 0; i < assets; ++i) {
        for (std::size_t j = i + 1; j < assets; ++j) {
            const double value = corr[i * assets + j];
            if (value >= 0.5) {
                adjacency[i * assets + j] = value;
                adjacency[j * assets + i] = value;
            }
        }
    }
    const auto network = rflux::percolab::connected_components(
        adjacency.data(), assets, 0.5);
    std::printf("components=%zu largest=%zu\n", network.n_components,
                network.largest_size);
    const auto strength =
        rflux::percolab::strength(adjacency.data(), assets);
    const auto centrality =
        rflux::percolab::eigenvector_centrality(adjacency.data(), assets);
    for (std::size_t a = 0; a < assets; ++a) {
        std::printf("asset %zu: strength=%.3f eigen=%.3f\n", a,
                    strength[a], centrality[a]);
    }
    const auto path =
        rflux::percolab::path_statistics(adjacency.data(), assets);
    std::printf(
        "paths: efficiency=%.3f average=%.3f diameter=%zu connected=%s\n",
        path.efficiency, path.average_path_length, path.diameter,
        path.connected ? "yes" : "no");
    const auto cores =
        rflux::percolab::core_numbers(adjacency.data(), assets);
    std::printf("core numbers:");
    for (std::int32_t value : cores) std::printf(" %d", value);
    std::printf(" | transitivity=%.3f\n",
                rflux::percolab::transitivity(adjacency.data(), assets));

    std::vector<std::uint32_t> removal_order(assets);
    for (std::size_t a = 0; a < assets; ++a) {
        removal_order[a] = static_cast<std::uint32_t>(a);
    }
    std::stable_sort(removal_order.begin(), removal_order.end(),
                     [&](std::uint32_t left, std::uint32_t right) {
                         return strength[left] > strength[right];
                     });
    const auto removal = rflux::percolab::targeted_removal(
        adjacency.data(), assets, removal_order.data(), 5);
    std::printf("targeted removal giant fraction:");
    for (const auto& point : removal) {
        std::printf(" %.2f", point.giant_fraction);
    }
    std::printf("\n");

    const std::vector<double> probabilities_scan = {0.0, 0.25, 0.5, 0.75,
                                                    1.0};
    const auto percolation = rflux::percolab::bond_percolation(
        adjacency.data(), assets, probabilities_scan.data(),
        probabilities_scan.size(), 200, 17);
    std::printf("bond percolation giant component:\n");
    for (const auto& point : percolation) {
        std::printf("  p=%.2f giant=%.3f susceptibility=%.3f\n",
                    point.probability, point.giant_mean,
                    point.susceptibility);
    }

    const std::uint32_t seed_nodes[1] = {0};
    const auto diffusion = rflux::percolab::risk_diffusion(
        adjacency.data(), assets, seed_nodes, 1, 2.0, 0.1, 20, 100, 17);
    std::printf("risk diffusion final affected=%.3f\n",
                diffusion.mean_affected.back());
    return 0;
}
