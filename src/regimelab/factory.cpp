#include "rflux/regimelab/factory.hpp"

#include <memory>

#include "rflux/regimelab/detector.hpp"

namespace rflux::regimelab {

void register_builtin_regime_detectors() {
    static const bool registered = [] {
        RegimeDetectorFactory::instance().add(
            "gaussian_hmm", [](const RegimeSpec& spec) {
                return std::make_unique<HmmRegimeDetector>(
                    spec.n_states, spec.n_features, spec.options);
            });
        return true;
    }();
    (void)registered;
}

}  // namespace rflux::regimelab
