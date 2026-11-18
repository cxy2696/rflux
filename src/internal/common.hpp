#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rflux {

/* Internal exception hierarchy.  The C ABI translates these to rflux_status.
 * The explicit default visibility keeps RTTI usable when the core is compiled
 * with LTO and linked into the pybind11 module: both sides must agree on one
 * std::type_info for the translator to catch the concrete type. */
#if defined(_WIN32)
#define RFLUX_EXCEPTION_VISIBILITY
#elif defined(__GNUC__) || defined(__clang__)
#define RFLUX_EXCEPTION_VISIBILITY __attribute__((visibility("default")))
#else
#define RFLUX_EXCEPTION_VISIBILITY
#endif

class RFLUX_EXCEPTION_VISIBILITY InvalidArgument : public std::invalid_argument {
public:
    explicit InvalidArgument(const std::string& what)
        : std::invalid_argument(what) {}
};

class RFLUX_EXCEPTION_VISIBILITY NonFinite : public InvalidArgument {
public:
    explicit NonFinite(const std::string& what) : InvalidArgument(what) {}
};

class RFLUX_EXCEPTION_VISIBILITY NotReady : public std::runtime_error {
public:
    explicit NotReady(const std::string& what) : std::runtime_error(what) {}
};

class RFLUX_EXCEPTION_VISIBILITY NumericalError : public std::runtime_error {
public:
    explicit NumericalError(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kLog2Pi = 1.837877066409345483560659472811235279;

inline void require(bool condition, const char* message) {
    if (!condition) throw InvalidArgument(message);
}

inline void require_pointer(const void* pointer, const char* message) {
    if (pointer == nullptr) throw InvalidArgument(message);
}

inline std::size_t checked_mul(std::size_t a, std::size_t b,
                               const char* message) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw InvalidArgument(message);
    }
    return a * b;
}

inline bool is_finite(double value) { return std::isfinite(value); }

inline void require_finite(const double* values, std::size_t n,
                           const char* what) {
    require_pointer(values, what);
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_finite(values[i])) {
            throw NonFinite(std::string(what) + " contains a non-finite value");
        }
    }
}

inline void require_probability(double value, const char* what) {
    if (!is_finite(value) || value < 0.0 || value > 1.0) {
        throw InvalidArgument(std::string(what) +
                              " must be a finite probability in [0, 1]");
    }
}

inline void require_positive(double value, const char* what) {
    if (!is_finite(value) || value <= 0.0) {
        throw InvalidArgument(std::string(what) + " must be finite and positive");
    }
}

inline double safe_log(double value) {
    return value > 0.0 ? std::log(value) : -std::numeric_limits<double>::infinity();
}

inline double log_add_exp(double a, double b) {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    const double high = std::max(a, b);
    const double low = std::min(a, b);
    return high + std::log1p(std::exp(low - high));
}

inline double log_sum_exp(const double* values, std::size_t n) {
    double high = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) high = std::max(high, values[i]);
    if (!is_finite(high)) return high;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += std::exp(values[i] - high);
    return high + std::log(sum);
}

/*
 * Type-7 quantile over an unsorted buffer.  Reorders `values` in place with
 * two order-statistic passes instead of a full sort, which is roughly three
 * times faster for large samples.  Only the two adjacent order statistics
 * (and, for q in {0, 1}, one extreme) are needed.
 */
inline double quantile_select(double* values, std::size_t n, double q) {
    if (n == 0) return kNaN;
    if (q <= 0.0) return *std::min_element(values, values + n);
    if (q >= 1.0) return *std::max_element(values, values + n);
    const double position = q * static_cast<double>(n - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1, n - 1);
    if (lower == upper) {
        std::nth_element(values, values + lower, values + n);
        return values[lower];
    }
    /* Partition at the upper order statistic; its prefix then holds exactly
     * the `upper` smallest values, so the lower statistic can be selected
     * inside that prefix without touching the right partition again. */
    std::nth_element(values, values + upper, values + n);
    const double high = values[upper];
    std::nth_element(values, values + lower, values + upper);
    const double low = values[lower];
    const double fraction = position - static_cast<double>(lower);
    return low + fraction * (high - low);
}

/* Type-7 linear interpolation over an already sorted vector. */
inline double quantile_sorted(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return kNaN;
    if (q <= 0.0) return sorted.front();
    if (q >= 1.0) return sorted.back();
    const double position = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

inline std::vector<double> finite_values(const double* values, std::size_t n) {
    require_pointer(values, "input");
    std::vector<double> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (is_finite(values[i])) result.push_back(values[i]);
    }
    return result;
}

inline double mean_of(const std::vector<double>& values) {
    if (values.empty()) return kNaN;
    long double sum = 0.0L;
    for (double value : values) sum += static_cast<long double>(value);
    return static_cast<double>(sum / static_cast<long double>(values.size()));
}

/* Population mean/variance with Welford's recurrence.  Keeping the update in
 * one place avoids the E[x^2] - E[x]^2 cancellation in replicated kernels. */
struct RunningStats {
    std::size_t count = 0;
    long double mean = 0.0L;
    long double m2 = 0.0L;

    void add(long double value) {
        ++count;
        const long double delta = value - mean;
        mean += delta / static_cast<long double>(count);
        m2 += delta * (value - mean);
    }

    long double variance() const {
        if (count == 0) return 0.0L;
        const long double value = m2 / static_cast<long double>(count);
        return value > 0.0L ? value : 0.0L;
    }
};

/* xoshiro256** seeded through SplitMix64.  Deterministic across platforms. */
class Rng {
public:
    explicit Rng(std::uint64_t seed) {
        std::uint64_t x = seed;
        for (std::uint64_t& word : state_) {
            x += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            word = z ^ (z >> 31);
        }
        if ((state_[0] | state_[1] | state_[2] | state_[3]) == 0) {
            state_[0] = 1;
        }
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(state_[1] * 5ULL, 7) * 9ULL;
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    /* Uniform double in [0, 1). */
    double uniform() {
        return static_cast<double>(next_u64() >> 11) *
               0x1.0p-53;
    }

    std::uint32_t uniform_index(std::uint32_t bound) {
        require(bound > 0, "uniform_index bound must be positive");
        return static_cast<std::uint32_t>(
            next_u64() % static_cast<std::uint64_t>(bound));
    }

    double normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u = 0.0;
        while (u <= 0.0) u = uniform();
        const double v = uniform();
        const double radius = std::sqrt(-2.0 * std::log(u));
        const double angle = 2.0 * kPi * v;
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    static std::uint64_t rotl(std::uint64_t value, int shift) {
        return (value << shift) | (value >> (64 - shift));
    }

    std::uint64_t state_[4]{};
    double spare_ = 0.0;
    bool has_spare_ = false;
};

inline std::uint64_t mix_seed(std::uint64_t seed, std::uint64_t index) {
    return seed ^ (index * 0xD1B54A32D192ED03ULL);
}

} /* namespace detail */
} /* namespace rflux */
