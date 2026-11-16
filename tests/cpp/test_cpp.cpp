#include "rflux/percolab/network.hpp"
#include "rflux/percolab/percolation.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/ranking.hpp"
#include "rflux/qflux/stats.hpp"
#include "rflux/qflux/window.hpp"
#include "rflux/regimelab/hmm.hpp"
#include "rflux/regimelab/regime.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,     \
                         #condition);                                        \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

bool approx(double actual, double expected, double tolerance = 1e-10) {
    if (std::isnan(actual) || std::isnan(expected)) return false;
    return std::abs(actual - expected) <= tolerance;
}

int test_qflux() {
    std::vector<double> prices(10);
    std::iota(prices.begin(), prices.end(), 1.0);
    const std::vector<double> sma =
        rflux::qflux::sma(prices.data(), prices.size(), 3);
    CHECK(std::isnan(sma[0]));
    CHECK(approx(sma[2], 2.0));
    CHECK(approx(sma.back(), 9.0));

    const std::vector<double> returns =
        rflux::qflux::returns(prices.data(), prices.size(), true);
    CHECK(std::isnan(returns[0]));
    CHECK(approx(returns[1], std::log(2.0)));

    const std::vector<double> stddev =
        rflux::qflux::rolling_std(prices.data(), prices.size(), 3, 1);
    CHECK(approx(stddev[2], 1.0));
    const std::vector<double> zscore =
        rflux::qflux::rolling_zscore(prices.data(), prices.size(), 2, 0);
    CHECK(approx(zscore.back(), 1.0));

    const std::vector<double> ema =
        rflux::qflux::ema(prices.data(), prices.size(), 3);
    CHECK(approx(ema[1], 1.5));
    CHECK(approx(ema[2], 2.25));

    const std::vector<double> slope =
        rflux::qflux::rolling_slope(prices.data(), prices.size(), 5);
    CHECK(approx(slope.back(), 1.0));
    std::vector<double> benchmark(10);
    std::iota(benchmark.begin(), benchmark.end(), 1.0);
    std::vector<double> asset(10);
    for (std::size_t i = 0; i < asset.size(); ++i) {
        asset[i] = 2.0 * benchmark[i] + 3.0;
    }
    const std::vector<double> beta = rflux::qflux::rolling_beta(
        asset.data(), benchmark.data(), asset.size(), 5);
    CHECK(approx(beta[4], 2.0));
    CHECK(approx(beta.back(), 2.0));
    const std::vector<double> cov = rflux::qflux::rolling_cov(
        asset.data(), benchmark.data(), asset.size(), 5, 0);
    CHECK(approx(cov[4], 2.0 * 2.0, 1e-12));
    const std::vector<double> r2 =
        rflux::qflux::rolling_r2(prices.data(), prices.size(), 5);
    CHECK(approx(r2.back(), 1.0));

    const auto bands = rflux::qflux::bollinger(prices.data(), prices.size(),
                                               prices.size(), 2.0);
    CHECK(approx(bands.middle.back(), 5.5));
    CHECK(approx(bands.upper.back(), 5.5 + 2.0 * std::sqrt(8.25)));

    const auto macd =
        rflux::qflux::macd(prices.data(), prices.size(), 3, 5, 2);
    CHECK(approx(macd.histogram.back(), macd.macd.back() - macd.signal.back()));

    std::vector<double> rsi_prices(20);
    std::iota(rsi_prices.begin(), rsi_prices.end(), 1.0);
    const std::vector<double> rsi =
        rflux::qflux::rsi(rsi_prices.data(), rsi_prices.size(), 14);
    CHECK(approx(rsi[14], 100.0));
    CHECK(approx(rsi.back(), 100.0));

    CHECK(!std::isnan(rflux::qflux::drawdown(prices.data(), prices.size())[0]));
    bool threw = false;
    try {
        (void)rflux::qflux::rolling_sum(prices.data(), prices.size(), 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    return 0;
}

int test_stats_ranking() {
    std::vector<double> values(10);
    std::iota(values.begin(), values.end(), 1.0);
    const rflux::qflux::Moments moments =
        rflux::qflux::describe(values.data(), values.size());
    CHECK(moments.n_valid == 10);
    CHECK(approx(moments.mean, 5.5));
    CHECK(approx(moments.variance, 55.0 / 6.0));
    CHECK(approx(moments.skewness, 0.0));
    CHECK(approx(moments.excess_kurtosis, -1.2));

    const double q = rflux::qflux::quantile(values.data(), values.size(), 0.25);
    CHECK(approx(q, 3.25));
    const double corr = rflux::qflux::pearson(values.data(), values.data(), 10);
    CHECK(approx(corr, 1.0));

    const std::vector<double> ranking =
        rflux::qflux::rank(values.data(), values.size());
    CHECK(approx(ranking[0], 1.0));
    const std::vector<double> z =
        rflux::qflux::zscore(values.data(), values.size(), 0);
    CHECK(approx(z[0], -1.5666989036));
    const std::vector<double> winsorized =
        rflux::qflux::winsorize(values.data(), values.size(), 0.1, 0.9);
    CHECK(approx(winsorized[0], 1.9));
    CHECK(approx(winsorized.back(), 9.1));

    std::vector<double> grouped = {1.0, 2.0, 10.0, 12.0};
    std::vector<std::int32_t> groups = {0, 0, 1, 1};
    const std::vector<double> neutral =
        rflux::qflux::neutralize(grouped.data(), grouped.size(), groups.data(),
                                2);
    CHECK(approx(neutral[0], -0.5));
    CHECK(approx(neutral[2], -1.0));

    const std::vector<double> tail_returns = {-0.1, -0.05, 0.0, 0.05, 0.1};
    const rflux::qflux::TailRisk tail =
        rflux::qflux::historical_var_cvar(tail_returns.data(),
                                          tail_returns.size(), 0.8);
    CHECK(approx(tail.var, 0.06));
    CHECK(approx(tail.cvar, 0.1));
    return 0;
}

int test_hmm() {
    rflux::regimelab::GaussianHMM model(2, 1);
    CHECK(!model.ready());
    bool threw = false;
    try {
        (void)model.parameters();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    rflux::regimelab::HmmParameters parameters;
    parameters.start_prob = {0.6, 0.4};
    parameters.transition = {0.9, 0.1, 0.2, 0.8};
    parameters.means = {-1.0, 1.0};
    parameters.variances = {1.0, 1.0};
    model.set_parameters(parameters);
    CHECK(model.ready());

    const std::vector<double> forecast =
        model.forecast(parameters.start_prob.data(), 2, 1);
    CHECK(approx(forecast[0], 0.6 * 0.9 + 0.4 * 0.2));
    CHECK(approx(forecast[1], 0.6 * 0.1 + 0.4 * 0.8));

    const std::vector<double> observations = {-1.2, -0.9, -1.1, 1.0, 1.2, 0.8};
    const std::vector<std::int32_t> path =
        model.viterbi(observations.data(), observations.size());
    CHECK(path.size() == observations.size());
    CHECK(path[0] == 0 && path[1] == 0 && path[2] == 0);
    CHECK(path[3] == 1 && path[4] == 1 && path[5] == 1);

    const std::vector<double> filtered =
        model.filtered(observations.data(), observations.size());
    CHECK(filtered.size() == observations.size() * 2);
    for (std::size_t t = 0; t < observations.size(); ++t) {
        CHECK(approx(filtered[t * 2] + filtered[t * 2 + 1], 1.0, 1e-9));
    }
    const std::vector<double> smoothed =
        model.smoothed(observations.data(), observations.size());
    CHECK(smoothed.size() == observations.size() * 2);

    std::vector<double> bad_observations = observations;
    bad_observations[2] = std::numeric_limits<double>::quiet_NaN();
    threw = false;
    try {
        (void)model.viterbi(bad_observations.data(), bad_observations.size());
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    rflux::regimelab::GaussianHMM fitted(2, 1);
    std::vector<double> generated(400);
    for (std::size_t i = 0; i < generated.size(); ++i) {
        generated[i] = (i < generated.size() / 2 ? -1.0 : 1.0) +
                       0.1 * std::sin(static_cast<double>(i) * 0.3);
    }
    rflux::regimelab::HmmOptions options;
    options.max_iter = 150;
    options.n_init = 2;
    options.seed = 11;
    fitted.fit(generated.data(), generated.size(), options);
    CHECK(fitted.ready());
    CHECK(std::isfinite(fitted.log_likelihood()));
    CHECK(fitted.bic() >= fitted.aic() - 1e-8);
    CHECK(fitted.parameters().means[0] < fitted.parameters().means[1]);
    CHECK(approx(fitted.score(generated.data(), generated.size()),
                 fitted.log_likelihood(), 1e-8));

    const std::vector<std::int32_t> fitted_path =
        fitted.viterbi(generated.data(), generated.size());
    std::size_t first = 0;
    std::size_t last = 0;
    for (std::size_t i = 0; i < generated.size() / 2; ++i) {
        first += fitted_path[i] == 0;
    }
    for (std::size_t i = generated.size() / 2; i < generated.size(); ++i) {
        last += fitted_path[i] == 1;
    }
    CHECK(first > 190);
    CHECK(last > 190);
    return 0;
}

int test_regimes() {
    std::vector<double> returns(300, 0.0);
    for (std::size_t i = 1; i < returns.size(); ++i) {
        returns[i] = (i < 150 ? 0.005 : 0.02) *
                     std::sin(static_cast<double>(i) * 0.3);
    }
    const auto volatility =
        rflux::regimelab::volatility_regime(returns.data(), returns.size(), 20,
                                            60, 0.25, 0.75);
    CHECK(volatility.labels.size() == returns.size());
    CHECK(volatility.labels[0] == -1);
    bool saw_valid = false;
    for (std::int32_t label : volatility.labels) {
        CHECK(label >= -1 && label <= 2);
        saw_valid = saw_valid || label >= 0;
    }
    CHECK(saw_valid);

    std::vector<double> prices(300);
    for (std::size_t i = 0; i < prices.size(); ++i) {
        prices[i] = 100.0 + 0.2 * static_cast<double>(i) +
                    std::sin(static_cast<double>(i) * 0.1);
    }
    const auto trend =
        rflux::regimelab::trend_regime(prices.data(), prices.size(), 20, 60, 0);
    CHECK(trend.labels.back() == 2);
    const std::vector<std::int32_t> composite =
        rflux::regimelab::composite_regime(volatility.labels.data(),
                                           trend.labels.data(),
                                           volatility.labels.size());
    for (std::size_t i = 0; i < composite.size(); ++i) {
        if (volatility.labels[i] < 0 || trend.labels[i] < 0) {
            CHECK(composite[i] == -1);
        } else {
            CHECK(composite[i] ==
                  std::max(volatility.labels[i], 2 - trend.labels[i]));
        }
    }
    return 0;
}

int test_graphs() {
    const std::vector<double> corr = {
        1.0, 0.9, 0.2, 0.1,
        0.9, 1.0, 0.3, 0.1,
        0.2, 0.3, 1.0, 0.8,
        0.1, 0.1, 0.8, 1.0};
    const rflux::percolab::ComponentSummary components =
        rflux::percolab::connected_components(corr.data(), 4, 0.5);
    CHECK(components.n_components == 2);
    CHECK(components.largest_size == 2);
    CHECK(components.labels[0] == components.labels[1]);
    CHECK(components.labels[0] != components.labels[2]);

    const std::vector<double> distance =
        rflux::percolab::correlation_to_distance(corr.data(), 4);
    CHECK(approx(distance[1], std::sqrt(0.2)));
    const std::vector<rflux::percolab::MstEdge> tree =
        rflux::percolab::minimum_spanning_tree(distance.data(), 4);
    CHECK(tree.size() == 3);
    for (const auto& edge : tree) {
        CHECK(edge.u < 4 && edge.v < 4);
        CHECK(edge.u != edge.v);
        CHECK(std::isfinite(edge.weight));
    }

    const std::vector<double> centrality =
        rflux::percolab::eigenvector_centrality(corr.data(), 4);
    const double norm = std::inner_product(centrality.begin(), centrality.end(),
                                           centrality.begin(), 0.0);
    CHECK(approx(norm, 1.0, 1e-8));
    const std::vector<double> strength =
        rflux::percolab::strength(corr.data(), 4);
    CHECK(approx(strength[0], 1.2));

    {
        const std::vector<double> p3 = {
            0, 1, 0,
            1, 0, 1,
            0, 1, 0};
        const std::vector<double> p3_centrality =
            rflux::percolab::eigenvector_centrality(p3.data(), 3);
        CHECK(approx(p3_centrality[0], 0.5, 1e-8));
        CHECK(approx(p3_centrality[1], std::sqrt(0.5), 1e-8));
        CHECK(approx(p3_centrality[2], 0.5, 1e-8));

        const std::vector<double> star = {
            0, 1, 1, 1,
            1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0};
        const std::vector<double> star_centrality =
            rflux::percolab::eigenvector_centrality(star.data(), 4);
        CHECK(approx(star_centrality[0], std::sqrt(0.5), 1e-8));
        for (std::size_t i = 1; i < 4; ++i) {
            CHECK(approx(star_centrality[i], 0.40824829046, 1e-8));
        }
    }

    {
        const std::vector<double> cycle = {
            0, 1, 0,
            0, 0, 1,
            1, 0, 0};
        const std::vector<double> ranks =
            rflux::percolab::pagerank(cycle.data(), 3, 0.85, 100, 1e-12);
        CHECK(approx(ranks[0], 1.0 / 3.0, 1e-10));
        CHECK(approx(ranks[1], 1.0 / 3.0, 1e-10));
        CHECK(approx(ranks[2], 1.0 / 3.0, 1e-10));

        const std::vector<double> chain = {
            0, 1, 0,
            0, 0, 1,
            0, 0, 0};
        const std::vector<double> chain_ranks =
            rflux::percolab::pagerank(chain.data(), 3, 0.85, 100, 1e-12);
        CHECK(chain_ranks[2] > chain_ranks[1]);
        CHECK(chain_ranks[1] > chain_ranks[0]);
        CHECK(approx(chain_ranks[0] + chain_ranks[1] + chain_ranks[2], 1.0,
                     1e-12));
    }

    const std::vector<double> path = {
        0, 1, 0,
        1, 0, 1,
        0, 1, 0};
    const std::vector<double> coefficients =
        rflux::percolab::clustering_coefficients(path.data(), 3);
    CHECK(std::isnan(coefficients[0]));
    CHECK(approx(coefficients[1], 0.0));
    CHECK(std::isnan(coefficients[2]));

    const std::vector<double> complete = {
        0, 1, 1, 1,
        1, 0, 1, 1,
        1, 1, 0, 1,
        1, 1, 1, 0};
    const std::vector<double> probabilities = {0.0, 1.0};
    const std::vector<rflux::percolab::PercolationPoint> bond =
        rflux::percolab::bond_percolation(complete.data(), 4,
                                          probabilities.data(), 2, 20, 5);
    CHECK(approx(bond[0].giant_mean, 0.25));
    CHECK(approx(bond[1].giant_mean, 1.0));
    CHECK(approx(bond[0].giant_std, 0.0, 1e-8));
    CHECK(approx(bond[0].susceptibility, 0.75));
    const std::vector<rflux::percolab::PercolationPoint> site =
        rflux::percolab::site_percolation(complete.data(), 4,
                                          probabilities.data(), 2, 20, 5);
    CHECK(approx(site[0].giant_mean, 0.0));
    CHECK(approx(site[1].giant_mean, 1.0));

    const std::uint32_t seeds[1] = {0};
    const rflux::percolab::DiffusionResult diffusion =
        rflux::percolab::risk_diffusion(complete.data(), 4, seeds, 1, 10.0,
                                        0.0, 3, 5, 9);
    CHECK(approx(diffusion.mean_affected[0], 0.25));
    CHECK(diffusion.mean_affected[1] > 0.9);
    CHECK(approx(diffusion.infection_frequency[0], 1.0));

    const std::vector<double> empty(16, 0.0);
    const std::vector<double> zero_centrality =
        rflux::percolab::eigenvector_centrality(empty.data(), 4);
    for (double value : zero_centrality) CHECK(value == 0.0);
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_qflux();
    failures += test_stats_ranking();
    failures += test_hmm();
    failures += test_regimes();
    failures += test_graphs();
    if (failures == 0) {
        std::printf("test_cpp: all C++ core tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_cpp: %d test group(s) failed\n", failures);
    return 1;
}
