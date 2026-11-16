#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "internal/common.hpp"

namespace rflux::detail {

inline void validate_symmetric_matrix(const double* weights, std::size_t n,
                                      const char* what) {
    require_pointer(weights, what);
    require(n > 0, "n must be positive");
    checked_mul(n, n, "n*n overflows size_t");
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            const double a = weights[i * n + j];
            const double b = weights[j * n + i];
            if (!is_finite(a) || !is_finite(b)) {
                throw NonFinite(std::string(what) +
                                " must contain only finite values");
            }
            const double scale =
                std::max(1.0, std::max(std::abs(a), std::abs(b)));
            if (std::abs(a - b) > 1e-9 * scale) {
                throw InvalidArgument(std::string(what) +
                                      " must be symmetric");
            }
        }
    }
}

inline void require_nonnegative_weights(const double* weights, std::size_t n,
                                        const char* what) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (weights[i * n + j] < 0.0) {
                throw InvalidArgument(std::string(what) +
                                      " must be non-negative");
            }
        }
    }
}

inline void require_adjacency(const double* weights, std::size_t n,
                              const char* what) {
    validate_symmetric_matrix(weights, n, what);
    require_nonnegative_weights(weights, n, what);
}

class UnionFind {
public:
    explicit UnionFind(std::size_t n) : parent_(n), size_(n, 1) {
        for (std::size_t i = 0; i < n; ++i) parent_[i] = i;
    }

    std::size_t find(std::size_t node) {
        std::size_t root = node;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[node] != node) {
            const std::size_t next = parent_[node];
            parent_[node] = root;
            node = next;
        }
        return root;
    }

    bool unite(std::size_t a, std::size_t b) {
        std::size_t root_a = find(a);
        std::size_t root_b = find(b);
        if (root_a == root_b) return false;
        if (size_[root_a] < size_[root_b]) std::swap(root_a, root_b);
        parent_[root_b] = root_a;
        size_[root_a] += size_[root_b];
        return true;
    }

    std::size_t component_size(std::size_t node) {
        return size_[find(node)];
    }

    std::size_t max_size() {
        std::size_t largest = 0;
        for (std::size_t i = 0; i < parent_.size(); ++i) {
            if (parent_[i] == i) largest = std::max(largest, size_[i]);
        }
        return largest;
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> size_;
};

/* Labels 0..C-1 in order of each component's smallest node. */
inline std::vector<std::int32_t> component_labels(UnionFind& uf,
                                                  std::size_t n,
                                                  std::size_t* n_components,
                                                  std::size_t* largest_size) {
    require(n > 0, "n must be positive");
    require(n <= static_cast<std::size_t>(
                     std::numeric_limits<std::int32_t>::max()),
            "n must fit in int32_t");
    std::vector<std::int32_t> labels(n, -1);
    std::vector<std::int32_t> root_label(n, -1);
    std::size_t next_label = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t root = uf.find(i);
        if (root_label[root] < 0) {
            root_label[root] = static_cast<std::int32_t>(next_label++);
        }
        labels[i] = root_label[root];
    }
    std::vector<std::size_t> counts(next_label, 0);
    for (std::int32_t label : labels) {
        ++counts[static_cast<std::size_t>(label)];
    }
    std::size_t largest = 0;
    for (std::size_t count : counts) largest = std::max(largest, count);
    *n_components = next_label;
    *largest_size = largest;
    return labels;
}

}  // namespace rflux::detail
