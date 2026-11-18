#include "rflux/qflux/ranking.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "internal/common.hpp"

namespace rflux::qflux {
namespace {

using rflux::detail::kNaN;
using rflux::detail::require;
using rflux::detail::require_pointer;

struct RankedValue {
    double value;
    std::size_t index;
};

std::vector<RankedValue> sorted_finite(const double* values, std::size_t n) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    std::vector<RankedValue> items;
    items.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) items.push_back({values[i], i});
    }
    std::sort(items.begin(), items.end(),
              [](const RankedValue& a, const RankedValue& b) {
                  if (a.value != b.value) return a.value < b.value;
                  return a.index < b.index;
              });
    return items;
}

}  // namespace

void rank_into(const double* values, std::size_t n, double* out) {
    const std::vector<RankedValue> items = sorted_finite(values, n);
    require_pointer(out, "out must not be NULL");
    std::fill(out, out + n, kNaN);
    std::size_t begin = 0;
    while (begin < items.size()) {
        std::size_t end = begin + 1;
        while (end < items.size() && items[end].value == items[begin].value) {
            ++end;
        }
        const double average_rank =
            (static_cast<double>(begin + 1) + static_cast<double>(end)) / 2.0;
        for (std::size_t k = begin; k < end; ++k) {
            out[items[k].index] = average_rank;
        }
        begin = end;
    }
}

std::vector<double> rank(const double* values, std::size_t n) {
    std::vector<double> out(n, kNaN);
    rank_into(values, n, out.data());
    return out;
}

void percentile_rank_into(const double* values, std::size_t n, double* out) {
    rank_into(values, n, out);
    std::size_t valid = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(out[i])) ++valid;
    }
    if (valid == 0) return;
    const double denominator = static_cast<double>(valid);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(out[i])) {
            out[i] = (out[i] - 0.5) / denominator;
        }
    }
}

std::vector<double> percentile_rank(const double* values, std::size_t n) {
    std::vector<double> out(n, kNaN);
    percentile_rank_into(values, n, out.data());
    return out;
}

void zscore_into(const double* values, std::size_t n, std::size_t ddof,
                 double* out) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(ddof <= 1, "ddof must be 0 or 1");
    require_pointer(out, "out must not be NULL");
    std::fill(out, out + n, kNaN);

    std::size_t valid = 0;
    long double sum = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            sum += static_cast<long double>(values[i]);
            ++valid;
        }
    }
    if (valid == 0 || valid <= ddof) return;
    const long double mean = sum / static_cast<long double>(valid);
    long double squared = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) continue;
        const long double diff = static_cast<long double>(values[i]) - mean;
        squared += diff * diff;
    }
    const long double denominator = static_cast<long double>(valid - ddof);
    const double standard_deviation =
        std::sqrt(static_cast<double>(squared / denominator));
    if (!(standard_deviation > 0.0)) return;
    const double center = static_cast<double>(mean);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            out[i] = (values[i] - center) / standard_deviation;
        }
    }
}

std::vector<double> zscore(const double* values, std::size_t n,
                           std::size_t ddof) {
    std::vector<double> out(n, kNaN);
    zscore_into(values, n, ddof, out.data());
    return out;
}

void winsorize_into(const double* values, std::size_t n, double lower_q,
                    double upper_q, double* out) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(std::isfinite(lower_q) && std::isfinite(upper_q) &&
                lower_q >= 0.0 && lower_q <= upper_q && upper_q <= 1.0,
            "quantiles must satisfy 0 <= lower_q <= upper_q <= 1");
    require_pointer(out, "out must not be NULL");

    std::vector<double> finite;
    finite.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) finite.push_back(values[i]);
    }
    std::fill(out, out + n, kNaN);
    if (finite.empty()) return;
    const double lower =
        rflux::detail::quantile_select(finite.data(), finite.size(), lower_q);
    const double upper =
        rflux::detail::quantile_select(finite.data(), finite.size(), upper_q);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            out[i] = std::max(lower, std::min(upper, values[i]));
        }
    }
}

std::vector<double> winsorize(const double* values, std::size_t n,
                              double lower_q, double upper_q) {
    std::vector<double> out(n, kNaN);
    winsorize_into(values, n, lower_q, upper_q, out.data());
    return out;
}

void neutralize_into(const double* values, std::size_t n,
                     const std::int32_t* groups, std::size_t group_count,
                     double* out) {
    require_pointer(values, "values must not be NULL");
    require_pointer(groups, "groups must not be NULL");
    require(n > 0, "n must be positive");
    require(group_count > 0, "group_count must be positive");
    require_pointer(out, "out must not be NULL");

    std::vector<long double> sums(group_count, 0.0L);
    std::vector<std::size_t> counts(group_count, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t group = groups[i];
        if (group == -1) continue;
        require(group >= 0 && static_cast<std::size_t>(group) < group_count,
                "groups must be -1 or in [0, group_count)");
        if (std::isfinite(values[i])) {
            sums[static_cast<std::size_t>(group)] +=
                static_cast<long double>(values[i]);
            ++counts[static_cast<std::size_t>(group)];
        }
    }

    std::fill(out, out + n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t group = groups[i];
        if (group == -1 || !std::isfinite(values[i])) continue;
        const std::size_t index = static_cast<std::size_t>(group);
        if (counts[index] == 0) continue;
        const double mean = static_cast<double>(
            sums[index] / static_cast<long double>(counts[index]));
        out[i] = values[i] - mean;
    }
}

std::vector<double> neutralize(const double* values, std::size_t n,
                               const std::int32_t* groups,
                               std::size_t group_count) {
    std::vector<double> out(n, kNaN);
    neutralize_into(values, n, groups, group_count, out.data());
    return out;
}

}  // namespace rflux::qflux
