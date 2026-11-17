#include "rflux/mllab/factory.hpp"

#include <memory>

#include "rflux/mllab/huber.hpp"

namespace rflux::mllab {

void register_builtin_predictors() {
    static const bool registered = [] {
        PredictorFactory::instance().add(
            "huber", [](const PredictorSpec& spec) {
                return std::make_unique<HuberRegressor>(
                    spec.epsilon, spec.max_iter, spec.tol, spec.fit_intercept);
            });
        return true;
    }();
    (void)registered;
}

}  // namespace rflux::mllab
