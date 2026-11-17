#include "rflux/percolab/factory.hpp"

#include <memory>

namespace rflux::percolab {

void register_builtin_percolation_detectors() {
    static const bool registered = [] {
        PercolationDetectorFactory::instance().add(
            "site_percolation", [](const PercolationSpec& spec) {
                SitePercolationDetector::Options options;
                options.repetitions = spec.repetitions;
                options.seed = spec.seed;
                options.crisis_fraction = spec.crisis_fraction;
                options.threshold_points = spec.threshold_points;
                return std::make_unique<SitePercolationDetector>(options);
            });
        return true;
    }();
    (void)registered;
}

}  // namespace rflux::percolab
