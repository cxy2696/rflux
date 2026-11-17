#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "internal/common.hpp"
#include "rflux/execution/almgren_chriss.hpp"
#include "rflux/execution/factory.hpp"
#include "rflux/core/market.hpp"
#include "rflux/risk/pretrade.hpp"
#include "rflux/factorlab/factors.hpp"
#include "rflux/mllab/huber.hpp"
#include "rflux/mllab/factory.hpp"
#include "rflux/regimelab/detector.hpp"
#include "rflux/regimelab/factory.hpp"
#include "rflux/portfoliolab/hrp.hpp"
#include "rflux/portfoliolab/factory.hpp"
#include "rflux/percolab/detector.hpp"
#include "rflux/percolab/factory.hpp"
#include "rflux/factorlab/factory.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/window.hpp"
#include "rflux/regimelab/hmm.hpp"
#include "rflux/regimelab/regime.hpp"

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

    rflux::regimelab::GaussianHMM first(2, 1);
    rflux::regimelab::GaussianHMM second(2, 1);
    first.fit(observations.data(), observations.size(), options);
    second.fit(observations.data(), observations.size(), options);

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

TEST(ExecutionGTest, LinearScheduleAtZeroRisk) {
    const rflux::execution::AlmgrenChrissPlanner planner(1.0, 0.0, 0.0, 0.0);
    const auto plan = planner.plan(10.0, 4, 1.0);
    ASSERT_EQ(plan.trades.size(), 4u);
    for (double trade : plan.trades) {
        EXPECT_NEAR(trade, 2.5, 1e-12);
    }
    EXPECT_NEAR(plan.expected_cost, 400.0, 1e-10);
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
