#pragma once

/*
 * Stable extension contracts for new rflux algorithms.
 *
 * These interfaces contain no file, network, allocation-owner or Python
 * types.  Implementations operate on caller-owned contiguous buffers and
 * live in the C++ core; language bindings and configuration are composition
 * layers outside this boundary.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rflux::core {

/* Row-major n_obs x n_assets panel. */
struct PanelView {
    const double* data = nullptr;
    std::size_t n_obs = 0;
    std::size_t n_assets = 0;

    [[nodiscard]] std::size_t size() const noexcept {
        return n_obs * n_assets;
    }
};

/*
 * Input consumed by a cross-sectional time-series factor.  Panels are
 * row-major; an implementation must return a new row-major n_obs x n_assets
 * panel and may leave undefined cells as NaN.
 */
struct FactorContext {
    PanelView close;
    PanelView volume;
    PanelView high;
    PanelView low;
};

class BaseAlgorithm {
public:
    virtual ~BaseAlgorithm() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::string_view version() const noexcept {
        return "1";
    }
};

class BaseFactor : public BaseAlgorithm {
public:
    [[nodiscard]] virtual std::vector<double> compute(
        const FactorContext& context) const = 0;
};

/*
 * A regime detector owns no implicit global state.  fit() may be called once;
 * states() and probabilities() are read-only afterwards.  Implementations must
 * document whether their outputs are causal.
 */
class BaseRegimeDetector : public BaseAlgorithm {
public:
    virtual void fit(const double* observations, std::size_t n_obs,
                     std::size_t n_features) = 0;
    [[nodiscard]] virtual std::vector<std::int32_t> states(
        const double* observations, std::size_t n_obs) const = 0;
    [[nodiscard]] virtual std::vector<double> probabilities(
        const double* observations, std::size_t n_obs) const = 0;
};

/* Long-only, unit-gross allocator. */
class BaseAllocator : public BaseAlgorithm {
public:
    [[nodiscard]] virtual std::vector<double> weights(
        const double* covariance, std::size_t n_assets) const = 0;
};

/* Linear or nonlinear predictor over a caller-owned feature matrix. */
class BasePredictor : public BaseAlgorithm {
public:
    virtual void fit(const double* features, const double* target,
                     std::size_t n_obs, std::size_t n_features) = 0;
    [[nodiscard]] virtual std::vector<double> predict(
        const double* features, std::size_t n_obs,
        std::size_t n_features) const = 0;
};

struct ExecutionPlan {
    std::vector<double> trades;  /* one entry per slice, summing to quantity */
    double expected_cost = 0.0;
    double expected_variance = 0.0;
    double kappa = 0.0;
};

class BaseExecutionPlanner : public BaseAlgorithm {
public:
    [[nodiscard]] virtual ExecutionPlan plan(double quantity,
                                             std::size_t slices,
                                             double horizon) const = 0;
};

namespace detail {

inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

inline void require_panel(const PanelView& panel, const char* name) {
    (void)name;
    require(panel.data != nullptr, "panel data must not be null");
    require(panel.n_obs > 0, "panel must have at least one observation");
    require(panel.n_assets > 0, "panel must have at least one asset");
    require(panel.n_assets <=
                std::numeric_limits<std::size_t>::max() / panel.n_obs,
            "panel dimensions overflow");
}

inline void require_same_shape(const PanelView& a, const PanelView& b,
                               const char* name) {
    (void)name;
    require(a.n_obs == b.n_obs && a.n_assets == b.n_assets,
            "panel shapes must match");
}

}  // namespace detail

}  // namespace rflux::core
