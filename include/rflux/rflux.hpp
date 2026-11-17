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
#include "rflux/percolab/detector.hpp"
#include "rflux/risk/pretrade.hpp"
#include "rflux/core/algorithm.hpp"
#include "rflux/core/market.hpp"
#include "rflux/core/registry.hpp"
#include "rflux/factorlab/factory.hpp"
#include "rflux/portfoliolab/factory.hpp"
#include "rflux/mllab/factory.hpp"
#include "rflux/execution/factory.hpp"
#include "rflux/percolab/factory.hpp"
#include "rflux/regimelab/factory.hpp"
#include "rflux/mllab/predictor.hpp"
#include "rflux/execution/execution_algo.hpp"
#include "rflux/factorlab/factors.hpp"
#include "rflux/portfoliolab/hrp.hpp"
#include "rflux/execution/almgren_chriss.hpp"
#include "rflux/mllab/huber.hpp"
#include "rflux/regimelab/detector.hpp"
