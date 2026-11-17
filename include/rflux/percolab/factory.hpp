#pragma once

#include <cstddef>
#include <cstdint>

#include "rflux/core/registry.hpp"
#include "rflux/percolab/detector.hpp"

namespace rflux::percolab {

struct PercolationSpec {
    std::size_t repetitions = 128;
    std::uint64_t seed = 42;
    double crisis_fraction = 0.5;
    std::size_t threshold_points = 21;
};

using PercolationDetectorFactory =
    core::AlgorithmRegistry<BasePercolationDetector, const PercolationSpec&>;

void register_builtin_percolation_detectors();

}  // namespace rflux::percolab
