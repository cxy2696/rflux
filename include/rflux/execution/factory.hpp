#pragma once

#include "rflux/core/algorithm.hpp"
#include "rflux/core/registry.hpp"

namespace rflux::execution {

struct ExecutionPlannerSpec {
    double temporary_impact = 0.0;
    double permanent_impact = 0.0;
    double volatility = 0.0;
    double risk_aversion = 0.0;
};

using ExecutionPlannerFactory = core::AlgorithmRegistry<
    core::BaseExecutionPlanner, const ExecutionPlannerSpec&>;

void register_builtin_execution_planners();

}  // namespace rflux::execution
