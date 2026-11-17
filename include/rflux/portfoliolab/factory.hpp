#pragma once

#include "rflux/core/algorithm.hpp"
#include "rflux/core/registry.hpp"

namespace rflux::portfoliolab {

using AllocatorFactory = core::AlgorithmRegistry<core::BaseAllocator>;

void register_builtin_allocators();

}  // namespace rflux::portfoliolab
