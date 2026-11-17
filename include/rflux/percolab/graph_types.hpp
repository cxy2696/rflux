#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::percolab {

/* Connected-component labels and summary shared by graph entry points. */
struct ComponentSummary {
    std::vector<std::int32_t> labels; /* 0..C-1 */
    std::size_t n_components = 0;
    std::size_t largest_size = 0;
};

}  // namespace rflux::percolab
