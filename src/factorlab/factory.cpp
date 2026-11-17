#include "rflux/factorlab/factory.hpp"

#include <memory>

#include "rflux/factorlab/factors.hpp"

namespace rflux::factorlab {

void register_builtin_factors() {
    static const bool registered = [] {
        FactorFactory::instance().add(
            "momentum", [](const FactorSpec& spec) {
                return std::make_unique<MomentumFactor>(spec.window);
            });
        FactorFactory::instance().add(
            "volume_reversal", [](const FactorSpec& spec) {
                return std::make_unique<VolumeReversalFactor>(spec.window);
            });
        FactorFactory::instance().add(
            "vwap_deviation", [](const FactorSpec& spec) {
                return std::make_unique<VwapDeviationFactor>(spec.window);
            });
        FactorFactory::instance().add(
            "amihud", [](const FactorSpec& spec) {
                return std::make_unique<AmihudFactor>(spec.window, spec.scale);
            });
        return true;
    }();
    (void)registered;
}

}  // namespace rflux::factorlab
