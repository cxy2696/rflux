#include "rflux/portfoliolab/factory.hpp"

#include <memory>

#include "rflux/portfoliolab/hrp.hpp"

namespace rflux::portfoliolab {

void register_builtin_allocators() {
    static const bool registered = [] {
        AllocatorFactory::instance().add(
            "hrp", [] { return std::make_unique<HRPAllocator>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace rflux::portfoliolab
