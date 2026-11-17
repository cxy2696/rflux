#pragma once

#include <cstddef>
#include <memory>

#include "rflux/core/algorithm.hpp"
#include "rflux/core/registry.hpp"

namespace rflux::factorlab {

/* Parameters consumed by the built-in factor constructors. */
struct FactorSpec {
    std::size_t window = 20;
    double scale = 1e6;
};

using FactorFactory =
    core::AlgorithmRegistry<core::BaseFactor, const FactorSpec&>;

/* Register all built-in factors.  Safe to call more than once. */
void register_builtin_factors();

/*
 * Macro for a factor implemented in one translation unit.  Example:
 *
 *   RFLUX_REGISTER_FACTOR("momentum",
 *       std::make_unique<MomentumFactor>(spec.window));
 *
 * The expression is evaluated inside a lambda receiving ``spec``.
 */
#define RFLUX_DETAIL_CONCAT_INNER(a, b) a##b
#define RFLUX_DETAIL_CONCAT(a, b) RFLUX_DETAIL_CONCAT_INNER(a, b)
#define RFLUX_REGISTER_FACTOR(NAME, EXPR)                                    \
    [[maybe_unused]] static const bool RFLUX_DETAIL_CONCAT(                  \
        rflux_factor_registration_, __COUNTER__) = [] {                     \
        FactorFactory::instance().add(                                       \
            NAME, [](const FactorSpec& spec) {                               \
                return std::unique_ptr<core::BaseFactor>(EXPR);              \
            });                                                              \
        return true;                                                         \
    }()

}  // namespace rflux::factorlab
