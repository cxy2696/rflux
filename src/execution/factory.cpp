#include "rflux/execution/factory.hpp"

#include <memory>

#include "rflux/execution/almgren_chriss.hpp"

namespace rflux::execution {

void register_builtin_execution_planners() {
    static const bool registered = [] {
        ExecutionPlannerFactory::instance().add(
            "almgren_chriss", [](const ExecutionPlannerSpec& spec) {
                return std::make_unique<AlmgrenChrissPlanner>(
                    spec.temporary_impact, spec.permanent_impact,
                    spec.volatility, spec.risk_aversion);
            });
        return true;
    }();
    (void)registered;
}

}  // namespace rflux::execution
