#pragma once

/*
 * Convenience umbrella for C++ users.  The individual headers remain the
 * authoritative declarations and can be included separately to reduce compile
 * time.  This header is a C++ source interface, not a stable C ABI.
 */

#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/ranking.hpp"
#include "rflux/qflux/stats.hpp"
#include "rflux/qflux/window.hpp"
#include "rflux/regimelab/hmm.hpp"
#include "rflux/regimelab/regime.hpp"
#include "rflux/percolab/network.hpp"
#include "rflux/percolab/percolation.hpp"
