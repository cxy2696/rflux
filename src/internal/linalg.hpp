#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "internal/common.hpp"

namespace rflux::detail {

/*
 * Solve a dense n x n system with partial pivoting.  The matrix and right-hand
 * side are passed by value so callers can keep their inputs.
 */
inline std::vector<long double> solve_dense(
    std::vector<long double> matrix, std::vector<long double> rhs,
    std::size_t n, const char* what) {
    require(n > 0, "linear system size must be positive");
    require(matrix.size() == n * n, "linear system matrix has wrong size");
    require(rhs.size() == n, "linear system right-hand side has wrong size");

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        long double best = std::abs(matrix[col * n + col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const long double value = std::abs(matrix[row * n + col]);
            if (value > best) {
                best = value;
                pivot = row;
            }
        }
        if (!(best > 1e-18L)) {
            throw NumericalError(std::string(what) +
                                 " is singular or has no unique solution");
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(matrix[col * n + j], matrix[pivot * n + j]);
            }
            std::swap(rhs[col], rhs[pivot]);
        }
        const long double diagonal = matrix[col * n + col];
        for (std::size_t row = col + 1; row < n; ++row) {
            const long double factor = matrix[row * n + col] / diagonal;
            if (factor == 0.0L) continue;
            for (std::size_t j = col; j < n; ++j) {
                matrix[row * n + j] -= factor * matrix[col * n + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    std::vector<long double> solution(n, 0.0L);
    for (std::size_t row = n; row-- > 0;) {
        long double value = rhs[row];
        for (std::size_t col = row + 1; col < n; ++col) {
            value -= matrix[row * n + col] * solution[col];
        }
        solution[row] = value / matrix[row * n + row];
    }
    return solution;
}

}  // namespace rflux::detail
