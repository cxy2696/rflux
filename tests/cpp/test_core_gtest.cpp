#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "internal/common.hpp"
#include "internal/parallel.hpp"
#include "rflux/execution/almgren_chriss.hpp"
#include "rflux/execution/factory.hpp"
#include "rflux/core/market.hpp"
#include "rflux/risk/pretrade.hpp"
#include "rflux/factorlab/evaluation.hpp"
#include "rflux/factorlab/factors.hpp"
#include "rflux/mllab/huber.hpp"
#include "rflux/mllab/factory.hpp"
#include "rflux/regimelab/detector.hpp"
#include "rflux/regimelab/factory.hpp"
#include "rflux/portfoliolab/hrp.hpp"
#include "rflux/portfoliolab/factory.hpp"
#include "rflux/percolab/detector.hpp"
#include "rflux/percolab/percolation.hpp"
#include "rflux/percolab/factory.hpp"
#include "rflux/factorlab/factory.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/ranking.hpp"
#include "rflux/qflux/stats.hpp"
#include "rflux/qflux/window.hpp"
#include "rflux/regimelab/hmm.hpp"
#include "rflux/regimelab/regime.hpp"
#include "rflux/risk/tail_dependence.hpp"

namespace {

bool close_enough(double actual, double expected, double tolerance = 1e-10) {
    return std::isfinite(actual) &&
           std::abs(actual - expected) <= tolerance;
}

std::vector<double> increasing(std::size_t n, double start = 1.0) {
    std::vector<double> values(n);
    std::iota(values.begin(), values.end(), start);
    return values;
}

rflux::regimelab::GaussianHMM fitted_model(
    const std::vector<double>& observations,
    const rflux::regimelab::HmmOptions& options) {
    rflux::regimelab::GaussianHMM model(2, 1);
    model.fit(observations.data(), observations.size(), options);
    return model;
}

/* Compare the buffer kernel with per-pair Pearson on contiguous columns. */
void expect_correlation_matrix_matches_pearson(
    const std::vector<double>& panel, std::size_t n_obs, std::size_t n_assets) {
    std::vector<double> matrix(n_assets * n_assets, 0.0);
    rflux::qflux::correlation_matrix_into(panel.data(), n_obs, n_assets,
                                          matrix.data());
    std::vector<std::vector<double>> columns(
        n_assets, std::vector<double>(n_obs, 0.0));
    for (std::size_t a = 0; a < n_assets; ++a) {
        for (std::size_t t = 0; t < n_obs; ++t) {
            columns[a][t] = panel[t * n_assets + a];
        }
    }
    for (std::size_t a = 0; a < n_assets; ++a) {
        EXPECT_NEAR(matrix[a * n_assets + a], 1.0, 1e-12);
        for (std::size_t b = 0; b < n_assets; ++b) {
            const double expected = rflux::qflux::pearson(
                columns[a].data(), columns[b].data(), n_obs);
            EXPECT_NEAR(matrix[a * n_assets + b], expected, 1e-12);
        }
    }
}

TEST(QfluxStatsGTest, DescribeRankingAndTailRisk) {
    std::vector<double> values(10);
    std::iota(values.begin(), values.end(), 1.0);
    const rflux::qflux::Moments moments =
        rflux::qflux::describe(values.data(), values.size());
    EXPECT_EQ(moments.n_valid, 10u);
    EXPECT_NEAR(moments.mean, 5.5, 1e-12);
    EXPECT_NEAR(moments.variance, 55.0 / 6.0, 1e-12);
    EXPECT_NEAR(moments.skewness, 0.0, 1e-12);
    EXPECT_NEAR(moments.excess_kurtosis, -1.2, 1e-12);

    EXPECT_NEAR(rflux::qflux::quantile(values.data(), values.size(), 0.25),
                3.25, 1e-12);
    EXPECT_NEAR(rflux::qflux::pearson(values.data(), values.data(),
                                      values.size()), 1.0, 1e-12);
    const std::vector<double> ranking =
        rflux::qflux::rank(values.data(), values.size());
    EXPECT_NEAR(ranking.front(), 1.0, 1e-12);
    const std::vector<double> z =
        rflux::qflux::zscore(values.data(), values.size(), 0);
    EXPECT_NEAR(z.front(), -1.5666989036, 1e-9);
    const std::vector<double> winsorized =
        rflux::qflux::winsorize(values.data(), values.size(), 0.1, 0.9);
    EXPECT_NEAR(winsorized.front(), 1.9, 1e-12);
    EXPECT_NEAR(winsorized.back(), 9.1, 1e-12);

    std::vector<double> grouped = {1.0, 2.0, 10.0, 12.0};
    std::vector<std::int32_t> groups = {0, 0, 1, 1};
    const std::vector<double> neutral =
        rflux::qflux::neutralize(grouped.data(), grouped.size(), groups.data(),
                                 2);
    EXPECT_NEAR(neutral[0], -0.5, 1e-12);
    EXPECT_NEAR(neutral[2], -1.0, 1e-12);

    const std::vector<double> tail_returns = {-0.1, -0.05, 0.0, 0.05, 0.1};
    const rflux::qflux::TailRisk tail =
        rflux::qflux::historical_var_cvar(tail_returns.data(),
                                          tail_returns.size(), 0.8);
    EXPECT_NEAR(tail.var, 0.06, 1e-12);
    EXPECT_NEAR(tail.cvar, 0.1, 1e-12);
}

TEST(QfluxGTest, RollingQuantileAndWarmup) {
    const std::vector<double> values = increasing(10);
    const auto median =
        rflux::qflux::rolling_quantile(values.data(), values.size(), 3, 0.5);
    ASSERT_EQ(median.size(), values.size());
    EXPECT_TRUE(std::isnan(median[0]));
    EXPECT_TRUE(std::isnan(median[1]));
    EXPECT_TRUE(close_enough(median[2], 2.0));
    EXPECT_TRUE(close_enough(median.back(), 9.0));
}

TEST(QfluxGTest, InvalidWindowThrows) {
    const std::vector<double> values = increasing(5);
    EXPECT_THROW(
        (void)rflux::qflux::rolling_sum(values.data(), values.size(), 0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)rflux::qflux::rolling_quantile(values.data(), values.size(), 2,
                                            1.5),
        std::invalid_argument);
}

TEST(RegimelabGTest, StructuralZerosSurviveForecast) {
    rflux::regimelab::GaussianHMM model(2, 1);
    rflux::regimelab::HmmParameters parameters;
    parameters.start_prob = {1.0, 0.0};
    parameters.transition = {1.0, 0.0, 0.0, 1.0};
    parameters.means = {-1.0, 1.0};
    parameters.variances = {1.0, 1.0};
    model.set_parameters(parameters);

    const auto forecast = model.forecast(parameters.start_prob.data(), 2, 50);
    ASSERT_EQ(forecast.size(), 2u);
    EXPECT_DOUBLE_EQ(forecast[0], 1.0);
    EXPECT_DOUBLE_EQ(forecast[1], 0.0);
}

TEST(RegimelabGTest, DeterministicFitForFixedSeed) {
    std::vector<double> observations(800);
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const double noise =
            std::sin(static_cast<double>(i) * 0.37) * 0.15;
        observations[i] = i < 400 ? -1.0 + noise : 1.0 + noise * 1.3;
    }
    rflux::regimelab::HmmOptions options;
    options.max_iter = 80;
    options.n_init = 2;
    options.seed = 17;

    const rflux::regimelab::GaussianHMM first = fitted_model(observations,
                                                             options);
    const rflux::regimelab::GaussianHMM second = fitted_model(observations,
                                                              options);

    EXPECT_EQ(first.iterations(), second.iterations());
    EXPECT_EQ(first.converged(), second.converged());
    EXPECT_DOUBLE_EQ(first.log_likelihood(), second.log_likelihood());
    EXPECT_EQ(first.parameters().means, second.parameters().means);

    const auto filtered =
        first.filtered(observations.data(), observations.size());
    ASSERT_EQ(filtered.size(), observations.size() * 2);
    for (std::size_t i = 0; i < observations.size(); ++i) {
        EXPECT_TRUE(close_enough(filtered[2 * i] + filtered[2 * i + 1], 1.0,
                                 1e-9));
    }
}

TEST(RegimelabGTest, FitRejectsNonFiniteObservations) {
    std::vector<double> observations = {0.0, 1.0, std::nan(""), 2.0};
    rflux::regimelab::GaussianHMM model(2, 1);
    EXPECT_THROW(model.fit(observations.data(), observations.size()),
                 rflux::NonFinite);
}

TEST(RegimelabGTest, RegimeLabelsAreCausal) {
    const std::vector<double> values = increasing(80, 1.0);
    const auto trend = rflux::regimelab::trend_regime(
        values.data(), values.size(), 5, 20, 0.0);
    ASSERT_EQ(trend.labels.size(), values.size());
    EXPECT_EQ(trend.labels[0], -1);
    EXPECT_EQ(trend.labels.back(), 2);
}

TEST(FactorLabGTest, InformationCoefficientAndTurnover) {
    const double factor[6] = {1, 2, 3, 4, 5, 6};
    const double forward[6] = {2, 4, 6, 8, 10, 12};
    const auto pearson =
        rflux::factorlab::information_coefficient(factor, forward, 2, 3, false);
    EXPECT_NEAR(pearson[0], 1.0, 1e-12);
    EXPECT_NEAR(pearson[1], 1.0, 1e-12);
    const auto spearman =
        rflux::factorlab::information_coefficient(factor, forward, 2, 3, true);
    EXPECT_NEAR(spearman[0], 1.0, 1e-12);
    const double weights[6] = {0.5, 0.5, 0.6, 0.4, 0.6, 0.4};
    const auto turnover = rflux::factorlab::factor_turnover(weights, 3, 2);
    EXPECT_NEAR(turnover[0], 0.0, 1e-12);
    EXPECT_NEAR(turnover[1], 0.1, 1e-12);
    EXPECT_NEAR(turnover[2], 0.0, 1e-12);
}

TEST(FactorLabGTest, MomentumAndVwapWarmup) {
    const double close[] = {1.0, 2.0, 3.0, 4.0};
    rflux::core::FactorContext context;
    context.close = {close, 4, 1};
    const rflux::factorlab::MomentumFactor momentum(2);
    const auto values = momentum.compute(context);
    ASSERT_EQ(values.size(), 4u);
    EXPECT_TRUE(std::isnan(values[0]));
    EXPECT_TRUE(std::isnan(values[1]));
    EXPECT_DOUBLE_EQ(values[2], 2.0);
    EXPECT_DOUBLE_EQ(values[3], 1.0);

    const double volume[] = {1.0, 1.0, 1.0, 2.0};
    context.volume = {volume, 4, 1};
    const rflux::factorlab::VwapDeviationFactor vwap(2);
    const auto deviation = vwap.compute(context);
    EXPECT_TRUE(std::isnan(deviation[0]));
    EXPECT_NEAR(deviation[1], 2.0 / 1.5 - 1.0, 1e-12);
    EXPECT_NEAR(deviation[2], 3.0 / 2.5 - 1.0, 1e-12);
    EXPECT_NEAR(deviation[3], 4.0 / ((3.0 * 1.0 + 4.0 * 2.0) / 3.0) - 1.0, 1e-12);
}

TEST(FactorLabGTest, VolumeReversalAndAmihud) {
    const double close[] = {1.0, 2.0, 4.0};
    const double volume[] = {1.0, 1.0, 2.0};
    rflux::core::FactorContext context;
    context.close = {close, 3, 1};
    context.volume = {volume, 3, 1};
    const rflux::factorlab::VolumeReversalFactor reversal(2);
    const auto values = reversal.compute(context);
    ASSERT_EQ(values.size(), 3u);
    EXPECT_TRUE(std::isnan(values[0]));
    EXPECT_TRUE(std::isnan(values[1]));
    EXPECT_NEAR(values[2], -4.0, 1e-12);

    const rflux::factorlab::AmihudFactor amihud(2, 1.0);
    const auto illiquidity = amihud.compute(context);
    ASSERT_EQ(illiquidity.size(), 3u);
    EXPECT_TRUE(std::isnan(illiquidity[0]));
    EXPECT_TRUE(std::isnan(illiquidity[1]));
    const double first = std::abs(2.0 / 1.0 - 1.0) / (2.0 * 1.0);
    const double second = std::abs(4.0 / 2.0 - 1.0) / (4.0 * 2.0);
    EXPECT_NEAR(illiquidity[2], (first + second) / 2.0, 1e-12);
}

TEST(PortfolioLabGTest, RiskParityAndNcoWeights) {
    const double covariance[4] = {0.04, 0.006, 0.006, 0.09};
    const auto parity =
        rflux::portfoliolab::risk_parity_weights(covariance, 2);
    EXPECT_NEAR(parity[0], 0.6923076923076923, 1e-12);
    EXPECT_NEAR(parity[1], 0.3076923076923077, 1e-12);
    const auto nco = rflux::portfoliolab::nco_weights(covariance, 2, 0.5);
    EXPECT_NEAR(nco[0] + nco[1], 1.0, 1e-12);
    EXPECT_GE(nco[0], 0.0);
    EXPECT_GE(nco[1], 0.0);
}

TEST(PortfolioLabGTest, RebalanceTargetsGateEachRow) {
    const double current[4] = {0.5, 0.5, 0.5, 0.5};
    const double target[4] = {0.51, 0.49, 0.7, 0.3};
    const auto gated =
        rflux::portfoliolab::rebalance_targets(current, target, 2, 2, 0.05);
    EXPECT_NEAR(gated[0], 0.5, 1e-12);
    EXPECT_NEAR(gated[1], 0.5, 1e-12);
    EXPECT_NEAR(gated[2], 0.7, 1e-12);
    EXPECT_NEAR(gated[3], 0.3, 1e-12);
}

TEST(RiskGTest, TailDependenceIsSymmetric) {
    const double returns[8] = {-2, -1, -1, 0, 0, 1, 1, 2};
    const auto lower =
        rflux::risk::tail_dependence(returns, 4, 2, 0.25, false);
    EXPECT_NEAR(lower[0], 1.0, 1e-12);
    EXPECT_NEAR(lower[3], 1.0, 1e-12);
    EXPECT_NEAR(lower[1], lower[2], 1e-12);
    EXPECT_GE(lower[1], 0.0);
    EXPECT_LE(lower[1], 1.0);
}

TEST(PortfolioLabGTest, HrpWeightsAreValid) {
    const double covariance[] = {
        0.04, 0.006,
        0.006, 0.09,
    };
    const rflux::portfoliolab::HRPAllocator allocator;
    const auto weights = allocator.weights(covariance, 2);
    ASSERT_EQ(weights.size(), 2u);
    EXPECT_GT(weights[0], 0.0);
    EXPECT_GT(weights[1], 0.0);
    EXPECT_NEAR(weights[0] + weights[1], 1.0, 1e-12);
    EXPECT_TRUE(std::isfinite(allocator.weights(covariance, 2)[0]));
}

TEST(ExecutionGTest, OrderBookMetrics) {
    const auto metrics = rflux::execution::order_book_metrics(
        (const double[]){99.9}, (const double[]){100.0},
        (const double[]){100.1}, (const double[]){200.0}, 1);
    EXPECT_NEAR(metrics.spread[0], 0.2, 1e-12);
    EXPECT_NEAR(metrics.mid[0], 100.0, 1e-12);
    EXPECT_NEAR(metrics.microprice[0], 99.9666666667, 1e-9);
    EXPECT_NEAR(metrics.imbalance[0], -1.0 / 3.0, 1e-12);
}

TEST(ExecutionGTest, LinearScheduleAtZeroRisk) {
    const rflux::execution::AlmgrenChrissPlanner planner(1.0, 0.0, 0.0, 0.0);
    const auto plan = planner.plan(10.0, 4, 1.0);
    ASSERT_EQ(plan.trades.size(), 4u);
    for (double trade : plan.trades) {
        EXPECT_NEAR(trade, 2.5, 1e-12);
    }
    EXPECT_NEAR(plan.expected_cost, 100.0, 1e-10);
    EXPECT_DOUBLE_EQ(plan.expected_variance, 0.0);
}

TEST(ExecutionGTest, RiskAversionFrontLoadsSchedule) {
    const rflux::execution::AlmgrenChrissPlanner planner(1.0, 0.1, 0.02, 2.0);
    const auto plan = planner.plan(1000.0, 10, 1.0);
    double total = 0.0;
    for (double trade : plan.trades) total += trade;
    EXPECT_NEAR(total, 1000.0, 1e-8);
    EXPECT_GT(plan.trades.front(), plan.trades.back());
    EXPECT_GT(plan.kappa, 0.0);
}

TEST(MLLabGTest, CvarAndGraphSmooth) {
    const double returns[4] = {0.01, -0.02, -0.05, 0.03};
    EXPECT_NEAR(rflux::mllab::cvar_loss(returns, 4, 0.95), 0.05, 1e-12);
    const double signal[3] = {1, 0, 1};
    const double adjacency[9] = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    const auto smooth =
        rflux::mllab::graph_smooth(signal, adjacency, 3, 0.5, 3);
    EXPECT_NEAR(smooth[0], smooth[2], 1e-12);
}

TEST(MLLabGTest, HuberResistsOutlier) {
    std::vector<double> features(100 * 2);
    std::vector<double> target(100);
    for (std::size_t i = 0; i < 100; ++i) {
        const double x0 = std::sin(static_cast<double>(i) * 0.31);
        const double x1 = std::cos(static_cast<double>(i) * 0.17);
        features[2 * i] = x0;
        features[2 * i + 1] = x1;
        target[i] = 1.0 + 2.0 * x0 - x1;
    }
    target[0] = 100.0;
    rflux::mllab::HuberRegressor regressor(1.35, 100, 1e-10);
    regressor.fit(features.data(), target.data(), 100, 2);
    EXPECT_TRUE(regressor.converged());
    EXPECT_NEAR(regressor.intercept(), 1.0, 1e-2);
    EXPECT_NEAR(regressor.coefficients()[0], 2.0, 1e-2);
    EXPECT_NEAR(regressor.coefficients()[1], -1.0, 1e-2);
    const auto prediction = regressor.predict(features.data(), 100, 2);
    EXPECT_EQ(prediction.size(), 100u);
}

TEST(RegimelabGTest, HmmRegimeDetectorAdapter) {
    std::vector<double> observations(300);
    for (std::size_t i = 0; i < observations.size(); ++i) {
        observations[i] = i < 150 ? -1.0 : 1.0;
    }
    rflux::regimelab::HmmOptions options;
    options.max_iter = 50;
    options.n_init = 2;
    options.seed = 7;
    rflux::regimelab::HmmRegimeDetector detector(2, 1, options);
    detector.fit(observations.data(), observations.size(), 1);
    const auto states = detector.states(observations.data(), observations.size());
    ASSERT_EQ(states.size(), observations.size());
    EXPECT_EQ(states[0], states[1]);
    EXPECT_NE(states[0], states.back());
    const auto probabilities =
        detector.probabilities(observations.data(), observations.size());
    ASSERT_EQ(probabilities.size(), observations.size() * 2);
    EXPECT_NEAR(probabilities[0] + probabilities[1], 1.0, 1e-9);
}

TEST(PercolationGTest, UnionFindSiteDetector) {
    const double adjacency[] = {
        0.0, 1.0,
        1.0, 0.0,
    };
    const auto network = rflux::percolab::make_network(adjacency, 2);
    EXPECT_EQ(network.num_nodes, 2u);
    EXPECT_EQ(network.edge_count(), 2u);
    rflux::percolab::SitePercolationDetector::Options options;
    options.repetitions = 256;
    options.threshold_points = 11;
    options.seed = 7;
    rflux::percolab::SitePercolationDetector detector(options);
    const auto zero = detector.analyze(network, 0.0);
    EXPECT_DOUBLE_EQ(zero.giant_component_fraction, 0.0);
    EXPECT_FALSE(zero.is_systemic_crisis);
    const auto full = detector.analyze(network, 1.0);
    EXPECT_DOUBLE_EQ(full.giant_component_fraction, 1.0);
    EXPECT_EQ(full.largest_component, 2u);
    const auto half = detector.analyze(network, 0.5);
    EXPECT_NEAR(half.giant_component_fraction, 0.5, 0.03);
    const double threshold = detector.estimate_threshold(network);
    EXPECT_GE(threshold, 0.0);
    EXPECT_LE(threshold, 1.0);
    EXPECT_TRUE(detector.calibrated());
    const auto calibrated = detector.analyze(network, 1.0);
    EXPECT_DOUBLE_EQ(calibrated.percolation_threshold, threshold);
    EXPECT_TRUE(calibrated.is_systemic_crisis);
    EXPECT_THROW((void)detector.analyze(network, 1.5),
                 std::invalid_argument);
}

TEST(FactorFactoryGTest, RegistersAndCreatesBuiltins) {
    rflux::factorlab::register_builtin_factors();
    auto& factory = rflux::factorlab::FactorFactory::instance();
    EXPECT_TRUE(factory.contains("momentum"));
    EXPECT_TRUE(factory.contains("amihud"));
    rflux::factorlab::FactorSpec spec;
    spec.window = 5;
    auto factor = factory.create("momentum", spec);
    ASSERT_NE(factor, nullptr);
    EXPECT_EQ(factor->name(), "momentum");
    EXPECT_THROW((void)factory.create("missing", spec),
                 std::invalid_argument);
    EXPECT_THROW(
        factory.add("momentum", [](const rflux::factorlab::FactorSpec&) {
            return std::unique_ptr<rflux::core::BaseFactor>();
        }),
        std::invalid_argument);
}

TEST(CoreMarketGTest, SnapshotValidation) {
    const double values[] = {1.0, 2.0};
    rflux::core::MarketSnapshotView snapshot;
    snapshot.n_assets = 2;
    snapshot.open = values;
    snapshot.high = values;
    snapshot.low = values;
    snapshot.close = values;
    snapshot.volume = values;
    EXPECT_NO_THROW(rflux::core::detail::require_snapshot(snapshot));
    snapshot.close = nullptr;
    EXPECT_THROW(rflux::core::detail::require_snapshot(snapshot),
                 std::invalid_argument);
}

TEST(RiskControlGTest, PositionDrawdownTurnoverAndPercolationGate) {
    rflux::core::RiskConstraints constraints;
    constraints.max_position_limit = 0.5;
    constraints.max_drawdown_limit = 0.2;
    constraints.turnover_threshold = 0.1;
    const rflux::risk::PreTradeRiskControl control(constraints);
    rflux::core::Order order;
    order.symbol = "600519.SH";
    order.quantity = 100;
    order.price = 10.0;
    EXPECT_TRUE(control.check(order, 0.4, 0.1, 0.05).allowed);
    EXPECT_FALSE(control.check(order, 0.6, 0.1, 0.05).allowed);
    EXPECT_FALSE(control.check(order, 0.4, 0.3, 0.05).allowed);
    EXPECT_FALSE(control.check(order, 0.4, 0.1, 0.2).allowed);

    rflux::percolab::PercolationResult crisis;
    crisis.is_systemic_crisis = true;
    EXPECT_FALSE(control.check(order, 0.4, 0.1, 0.05, &crisis).allowed);
    order.quantity = 0;
    EXPECT_FALSE(control.check(order, 0.4, 0.1, 0.05).allowed);
}

TEST(AlgorithmFactoriesGTest, CreateEveryBuiltinFamily) {
    rflux::portfoliolab::register_builtin_allocators();
    auto allocator = rflux::portfoliolab::AllocatorFactory::instance()
                         .create("hrp");
    ASSERT_NE(allocator, nullptr);
    EXPECT_EQ(allocator->name(), "hrp");

    rflux::mllab::register_builtin_predictors();
    rflux::mllab::PredictorSpec predictor_spec;
    auto predictor = rflux::mllab::PredictorFactory::instance().create(
        "huber", predictor_spec);
    ASSERT_NE(predictor, nullptr);
    EXPECT_EQ(predictor->name(), "huber_regression");

    rflux::execution::register_builtin_execution_planners();
    rflux::execution::ExecutionPlannerSpec execution_spec;
    execution_spec.temporary_impact = 1.0;
    auto planner = rflux::execution::ExecutionPlannerFactory::instance()
                       .create("almgren_chriss", execution_spec);
    ASSERT_NE(planner, nullptr);
    EXPECT_EQ(planner->name(), "almgren_chriss");

    rflux::percolab::register_builtin_percolation_detectors();
    rflux::percolab::PercolationSpec percolation_spec;
    auto detector = rflux::percolab::PercolationDetectorFactory::instance()
                        .create("site_percolation", percolation_spec);
    ASSERT_NE(detector, nullptr);
    EXPECT_EQ(detector->name(), "site_percolation");

    rflux::regimelab::register_builtin_regime_detectors();
    rflux::regimelab::RegimeSpec regime_spec;
    auto regime = rflux::regimelab::RegimeDetectorFactory::instance().create(
        "gaussian_hmm", regime_spec);
    ASSERT_NE(regime, nullptr);
    EXPECT_EQ(regime->name(), "gaussian_hmm");
}

}  // namespace

/* ------------------------------------------------------------------ */
/* Buffer-first kernels and deterministic parallel kernels             */
/* ------------------------------------------------------------------ */

/* Build a synthetic n_obs x n_assets panel from a value(t, a) callable. */
template <typename Value>
std::vector<double> synthetic_panel(std::size_t n_obs, std::size_t n_assets,
                                    Value&& value) {
    std::vector<double> panel(n_obs * n_assets);
    for (std::size_t t = 0; t < n_obs; ++t) {
        for (std::size_t a = 0; a < n_assets; ++a) {
            panel[t * n_assets + a] = value(t, a);
        }
    }
    return panel;
}

TEST(QfluxBufferGTest, BufferMatchesVectorOverloads) {
    std::vector<double> values(257);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double wave = std::sin(static_cast<double>(i) * 0.31) * 4.0;
        values[i] = (i % 41 == 7) ? std::nan("") : 50.0 + wave + 0.01 * i;
    }
    const std::size_t n = values.size();
    std::vector<double> other(n);
    for (std::size_t i = 0; i < n; ++i) {
        other[i] = (i % 37 == 5) ? std::nan("")
                                 : std::cos(static_cast<double>(i) * 0.17);
    }
    const auto check_vector = [&](const char* name, const double* expected,
                                  const double* actual) {
        for (std::size_t i = 0; i < n; ++i) {
            const bool both_nan =
                std::isnan(expected[i]) && std::isnan(actual[i]);
            ASSERT_TRUE(both_nan ||
                        std::abs(expected[i] - actual[i]) <= 1e-12)
                << name << " index " << i << " differs";
        }
    };
    std::vector<double> out(n, -1.0);

    for (std::size_t window : {1u, 7u, 64u}) {
        rflux::qflux::rolling_sum_into(values.data(), n, window, out.data());
        check_vector("rolling_sum",
                     rflux::qflux::rolling_sum(values.data(), n, window).data(),
                     out.data());
        rflux::qflux::rolling_std_into(values.data(), n, window, 0,
                                       out.data());
        check_vector("rolling_std",
                     rflux::qflux::rolling_std(values.data(), n, window, 0)
                         .data(),
                     out.data());
        rflux::qflux::rolling_zscore_into(values.data(), n, window, 0,
                                          out.data());
        check_vector("rolling_zscore",
                     rflux::qflux::rolling_zscore(values.data(), n, window, 0)
                         .data(),
                     out.data());
        rflux::qflux::rolling_min_into(values.data(), n, window, out.data());
        check_vector("rolling_min",
                     rflux::qflux::rolling_min(values.data(), n, window).data(),
                     out.data());
        rflux::qflux::rolling_max_into(values.data(), n, window, out.data());
        check_vector("rolling_max",
                     rflux::qflux::rolling_max(values.data(), n, window).data(),
                     out.data());
        rflux::qflux::rolling_quantile_into(values.data(), n, window, 0.3,
                                            out.data());
        check_vector("rolling_quantile",
                     rflux::qflux::rolling_quantile(values.data(), n, window,
                                                    0.3)
                         .data(),
                     out.data());
    }

    for (std::size_t window : {2u, 7u, 64u}) {
        rflux::qflux::rolling_corr_into(values.data(), other.data(), n, window,
                                        out.data());
        check_vector("rolling_corr",
                     rflux::qflux::rolling_corr(values.data(), other.data(), n,
                                                window)
                         .data(),
                     out.data());
        rflux::qflux::rolling_beta_into(values.data(), other.data(), n, window,
                                        out.data());
        check_vector("rolling_beta",
                     rflux::qflux::rolling_beta(values.data(), other.data(), n,
                                                window)
                         .data(),
                     out.data());
        rflux::qflux::rolling_slope_into(values.data(), n, window, out.data());
        check_vector("rolling_slope",
                     rflux::qflux::rolling_slope(values.data(), n, window).data(),
                     out.data());
        rflux::qflux::rolling_r2_into(values.data(), n, window, out.data());
        check_vector("rolling_r2",
                     rflux::qflux::rolling_r2(values.data(), n, window).data(),
                     out.data());
    }

    rflux::qflux::returns_into(values.data(), n, true, out.data());
    check_vector("returns", rflux::qflux::returns(values.data(), n, true).data(),
                 out.data());
    rflux::qflux::sma_into(values.data(), n, 5, out.data());
    check_vector("sma", rflux::qflux::sma(values.data(), n, 5).data(),
                 out.data());
    rflux::qflux::ema_into(values.data(), n, 5, out.data());
    check_vector("ema", rflux::qflux::ema(values.data(), n, 5).data(),
                 out.data());
    rflux::qflux::rsi_into(values.data(), n, 5, out.data());
    check_vector("rsi", rflux::qflux::rsi(values.data(), n, 5).data(),
                 out.data());
    rflux::qflux::drawdown_into(values.data(), n, out.data());
    check_vector("drawdown", rflux::qflux::drawdown(values.data(), n).data(),
                 out.data());
    rflux::qflux::atr_into(values.data(), other.data(), values.data(), n, 5,
                           out.data());
    check_vector("atr",
                 rflux::qflux::atr(values.data(), other.data(), values.data(),
                                   n, 5)
                     .data(),
                 out.data());

    const rflux::qflux::MacdResult macd_ref =
        rflux::qflux::macd(values.data(), n, 3, 6, 2);
    std::vector<double> macd_out(n, 0.0), signal_out(n, 0.0),
        hist_out(n, 0.0);
    rflux::qflux::macd_into(values.data(), n, 3, 6, 2, macd_out.data(),
                            signal_out.data(), hist_out.data());
    check_vector("macd", macd_ref.macd.data(), macd_out.data());
    check_vector("macd_signal", macd_ref.signal.data(), signal_out.data());
    check_vector("macd_histogram", macd_ref.histogram.data(), hist_out.data());

    const auto bands = rflux::qflux::bollinger(values.data(), n, 9, 2.0);
    std::vector<double> middle(n, 0.0), upper(n, 0.0), lower(n, 0.0);
    rflux::qflux::bollinger_into(values.data(), n, 9, 2.0, middle.data(),
                                 upper.data(), lower.data());
    check_vector("bollinger_middle", bands.middle.data(), middle.data());
    check_vector("bollinger_upper", bands.upper.data(), upper.data());
    check_vector("bollinger_lower", bands.lower.data(), lower.data());

    rflux::qflux::rank_into(values.data(), n, out.data());
    check_vector("rank", rflux::qflux::rank(values.data(), n).data(),
                 out.data());
    rflux::qflux::percentile_rank_into(values.data(), n, out.data());
    check_vector("percentile_rank",
                 rflux::qflux::percentile_rank(values.data(), n).data(),
                 out.data());
    rflux::qflux::zscore_into(values.data(), n, 1, out.data());
    check_vector("zscore", rflux::qflux::zscore(values.data(), n, 1).data(),
                 out.data());
    rflux::qflux::winsorize_into(values.data(), n, 0.05, 0.95, out.data());
    check_vector("winsorize",
                 rflux::qflux::winsorize(values.data(), n, 0.05, 0.95).data(),
                 out.data());
    std::vector<std::int32_t> groups(n);
    for (std::size_t i = 0; i < n; ++i) {
        groups[i] = static_cast<std::int32_t>(i % 5);
    }
    rflux::qflux::neutralize_into(values.data(), n, groups.data(), 5,
                                  out.data());
    check_vector("neutralize",
                 rflux::qflux::neutralize(values.data(), n, groups.data(), 5)
                     .data(),
                 out.data());
}

TEST(QfluxBufferGTest, RollingQuantileMatchesBruteForce) {
    std::vector<double> values(1400);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double wave = std::sin(static_cast<double>(i) * 0.7) * 3.0;
        values[i] = (i % 53 == 11) ? std::nan("") : 10.0 + wave;
    }
    for (std::size_t window : {1u, 3u, 64u, 1025u}) {
        for (double q : {0.0, 0.1, 0.5, 0.9, 1.0}) {
            const auto actual = rflux::qflux::rolling_quantile(
                values.data(), values.size(), window, q);
            for (std::size_t i = 0; i + 1 < window; ++i) {
                ASSERT_TRUE(std::isnan(actual[i]));
            }
            for (std::size_t i = window - 1; i < values.size(); ++i) {
                const std::size_t begin = i + 1 - window;
                std::vector<double> slice(values.begin() +
                                              static_cast<std::ptrdiff_t>(begin),
                                          values.begin() +
                                              static_cast<std::ptrdiff_t>(i + 1));
                const bool has_nan = std::any_of(slice.begin(), slice.end(),
                                                 [](double v) {
                                                     return !std::isfinite(v);
                                                 });
                if (has_nan) {
                    ASSERT_TRUE(std::isnan(actual[i]));
                    continue;
                }
                std::sort(slice.begin(), slice.end());
                const double position = q * static_cast<double>(window - 1);
                const std::size_t lower =
                    static_cast<std::size_t>(std::floor(position));
                const std::size_t upper = std::min(lower + 1, window - 1);
                const double expected =
                    slice[lower] + (position - static_cast<double>(lower)) *
                                       (slice[upper] - slice[lower]);
                ASSERT_NEAR(actual[i], expected, 1e-12);
            }
        }
    }
}

TEST(QfluxBufferGTest, BufferDoesNotWritePastLength) {
    const std::vector<double> values(37, 2.0);
    const double sentinel = 12345.0;
    std::vector<double> guarded(values.size() + 2, sentinel);
    rflux::qflux::rolling_std_into(values.data(), values.size(), 5, 0,
                                   guarded.data() + 1);
    rflux::qflux::rolling_min_into(values.data(), values.size(), 5,
                                   guarded.data() + 1);
    EXPECT_DOUBLE_EQ(guarded.front(), sentinel);
    EXPECT_DOUBLE_EQ(guarded.back(), sentinel);

    std::vector<double> macd(values.size() + 2, sentinel);
    std::vector<double> signal(values.size() + 2, sentinel);
    std::vector<double> histogram(values.size() + 2, sentinel);
    rflux::qflux::macd_into(values.data(), values.size(), 3, 7, 2,
                            macd.data() + 1, signal.data() + 1,
                            histogram.data() + 1);
    EXPECT_DOUBLE_EQ(macd.front(), sentinel);
    EXPECT_DOUBLE_EQ(signal.back(), sentinel);
    EXPECT_DOUBLE_EQ(histogram.front(), sentinel);
}

TEST(QfluxBufferGTest, CorrelationMatrixCompleteMatchesPearson) {
    const std::size_t n_obs = 64;
    const std::size_t n_assets = 6;
    const std::vector<double> panel = synthetic_panel(
        n_obs, n_assets, [](std::size_t t, std::size_t a) {
            return std::sin(static_cast<double>(t) * (0.1 + 0.05 * a)) +
                   0.01 * static_cast<double>(t * a);
        });
    expect_correlation_matrix_matches_pearson(panel, n_obs, n_assets);
}

TEST(RegimelabGTest, RestartsAreDeterministicAndOrdered) {
    std::vector<double> observations(1200);
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const double noise = std::sin(static_cast<double>(i) * 0.23) * 0.2;
        observations[i] = i < 600 ? -0.8 + noise : 0.9 + noise;
    }
    rflux::regimelab::HmmOptions options;
    options.max_iter = 60;
    options.n_init = 3;
    options.seed = 23;

    const rflux::regimelab::GaussianHMM first = fitted_model(observations,
                                                             options);
    const rflux::regimelab::GaussianHMM second = fitted_model(observations,
                                                              options);
    EXPECT_DOUBLE_EQ(first.log_likelihood(), second.log_likelihood());
    EXPECT_EQ(first.parameters().means, second.parameters().means);
    EXPECT_EQ(first.parameters().variances, second.parameters().variances);

    /* Restart 0 uses options.seed itself, so one restart reproduces that
     * candidate and cannot beat the multi-restart best. */
    rflux::regimelab::HmmOptions single = options;
    single.n_init = 1;
    const rflux::regimelab::GaussianHMM one = fitted_model(observations,
                                                           single);
    EXPECT_GE(first.log_likelihood(), one.log_likelihood());
}

TEST(RegimelabGTest, VolatilityRegimeMatchesFullSortReference) {
    const std::size_t n = 180;
    std::vector<double> returns(n);
    for (std::size_t i = 0; i < n; ++i) {
        returns[i] = (i % 97 == 0)
                         ? std::nan("")
                         : std::sin(static_cast<double>(i) * 0.37) * 0.01 +
                               0.002 * static_cast<double>(i % 5);
    }
    const std::size_t vol_window = 10;
    const std::size_t lookback = 40;
    const double low_q = 0.2;
    const double high_q = 0.8;
    const auto result = rflux::regimelab::volatility_regime(
        returns.data(), n, vol_window, lookback, low_q, high_q);

    std::deque<double> history;
    for (std::size_t t = 0; t < n; ++t) {
        const double value = result.volatility[t];
        if (!std::isfinite(value)) continue;
        std::int32_t expected = -1;
        if (history.size() >= 20) {
            std::vector<double> reference(history.begin(), history.end());
            std::sort(reference.begin(), reference.end());
            const double lower = rflux::detail::quantile_sorted(reference, low_q);
            const double upper = rflux::detail::quantile_sorted(reference, high_q);
            expected = value > upper ? 2 : (value < lower ? 0 : 1);
        }
        EXPECT_EQ(result.labels[t], expected) << "index " << t;
        history.push_back(value);
        if (history.size() > lookback) history.pop_front();
    }
}

TEST(PercolabGTest, ThresholdInterpolatesTheGiantCrossing) {
    const double adjacency[4] = {
        0.0, 1.0,
        1.0, 0.0,
    };
    const auto network = rflux::percolab::make_network(adjacency, 2);
    rflux::percolab::SitePercolationDetector::Options options;
    options.repetitions = 512;
    options.threshold_points = 6;  /* grid 0, .2, .4, .6, .8, 1 */
    options.crisis_fraction = 0.5;
    options.seed = 7;
    rflux::percolab::SitePercolationDetector detector(options);
    const double threshold = detector.estimate_threshold(network);
    /* For a two-node connected graph the mean giant fraction equals p, so a
     * grid scan first crosses 0.5 at p=0.6 while interpolation reaches 0.5. */
    EXPECT_NEAR(threshold, 0.5, 0.08);
    EXPECT_TRUE(detector.calibrated());
    EXPECT_DOUBLE_EQ(detector.analyze(network, 1.0).percolation_threshold,
                     threshold);
}

TEST(PercolabGTest, MonteCarloScansAreDeterministic) {
    /* n >= 128 also exercises the parallel scan path under TSan/ASan. */
    const std::size_t n = 160;
    std::vector<double> adjacency(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if ((i * 7 + j * 3) % 5 != 0) {
                const double weight =
                    0.2 + 0.8 * static_cast<double>((i * 13 + j * 11) % 17) /
                              16.0;
                adjacency[i * n + j] = weight;
                adjacency[j * n + i] = weight;
            }
        }
    }
    const double probabilities[] = {0.1, 0.3, 0.5, 0.7, 0.9};
    const auto first = rflux::percolab::site_percolation(
        adjacency.data(), n, probabilities, 5, 9, 123);
    const auto second = rflux::percolab::site_percolation(
        adjacency.data(), n, probabilities, 5, 9, 123);
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_DOUBLE_EQ(first[i].giant_mean, second[i].giant_mean);
        EXPECT_DOUBLE_EQ(first[i].giant_std, second[i].giant_std);
        EXPECT_DOUBLE_EQ(first[i].susceptibility, second[i].susceptibility);
    }
    const auto sizes_first = rflux::percolab::cluster_size_distribution(
        adjacency.data(), n, rflux::percolab::PercolationKind::site,
        rflux::percolab::OccupationWeights{}, 0.4, 8, 77);
    const auto sizes_second = rflux::percolab::cluster_size_distribution(
        adjacency.data(), n, rflux::percolab::PercolationKind::site,
        rflux::percolab::OccupationWeights{}, 0.4, 8, 77);
    ASSERT_EQ(sizes_first.size(), sizes_second.size());
    for (std::size_t i = 0; i < sizes_first.size(); ++i) {
        EXPECT_DOUBLE_EQ(sizes_first[i], sizes_second[i]);
    }
}

TEST(QfluxBufferGTest, CorrelationMatrixParallelPath) {
    const std::size_t n_obs = 8000;
    const std::size_t n_assets = 9;
    const std::vector<double> panel = synthetic_panel(
        n_obs, n_assets, [](std::size_t t, std::size_t a) {
            return std::sin(static_cast<double>(t) * (0.03 + 0.02 * a)) +
                   0.001 * static_cast<double>(t % (a + 3));
        });
    expect_correlation_matrix_matches_pearson(panel, n_obs, n_assets);
}

TEST(ParallelGTest, StaticPartitionAndExceptionPropagation) {
    std::vector<int> values(1000, 0);
    rflux::detail::parallel_for(
        values.size(),
        [&](std::size_t i) { values[i] = static_cast<int>(i * 2); }, 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], static_cast<int>(i * 2));
    }

    EXPECT_THROW(
        rflux::detail::parallel_for(
            200,
            [](std::size_t i) {
                if (i == 57) throw std::runtime_error("worker failure");
            },
            4),
        std::runtime_error);

    /* Nested calls run inline but still cover every index exactly once. */
    std::vector<int> nested(20, 0);
    rflux::detail::parallel_for(
        4,
        [&](std::size_t outer) {
            rflux::detail::parallel_for(
                5, [&](std::size_t inner) { nested[outer * 5 + inner] = 1; },
                4);
        },
        4);
    for (int value : nested) EXPECT_EQ(value, 1);
}

TEST(QfluxBufferGTest, QuantileMatchesSortedReference) {
    std::vector<double> values(513);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<double>((i * 37) % 53) - 26.0;
    }
    values[11] = std::nan("");
    values[400] = std::nan("");
    std::vector<double> finite;
    for (double value : values) {
        if (std::isfinite(value)) finite.push_back(value);
    }
    std::sort(finite.begin(), finite.end());
    for (double q : {0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0}) {
        const double position = q * static_cast<double>(finite.size() - 1);
        const std::size_t lower =
            static_cast<std::size_t>(std::floor(position));
        const std::size_t upper = std::min(lower + 1, finite.size() - 1);
        const double expected =
            finite[lower] + (position - static_cast<double>(lower)) *
                                (finite[upper] - finite[lower]);
        EXPECT_NEAR(rflux::qflux::quantile(values.data(), values.size(), q),
                    expected, 1e-12)
            << "q=" << q;
    }
}
