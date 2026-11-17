#include "rflux/portfoliolab/hrp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "internal/common.hpp"

namespace rflux::portfoliolab {
namespace {

using rflux::detail::kNaN;
using rflux::detail::require;

struct Node {
    std::size_t left = 0;
    std::size_t right = 0;
    std::size_t leaf = 0;
    bool is_leaf = false;
};

void validate_covariance(const double* covariance, std::size_t n) {
    rflux::detail::require_pointer(covariance, "covariance must not be null");
    require(n > 0, "covariance must not be empty");
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double value = covariance[i * n + j];
            require(std::isfinite(value), "covariance must be finite");
            require(std::abs(value - covariance[j * n + i]) <=
                        1e-8 * (1.0 + std::abs(value)),
                    "covariance must be symmetric");
        }
        require(covariance[i * n + i] > 0.0,
                "covariance diagonal must be positive");
    }
}

std::vector<std::size_t> cluster_order(const double* covariance,
                                       std::size_t n) {
    validate_covariance(covariance, n);
    std::vector<double> distance(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double corr = std::clamp(
                covariance[i * n + j] /
                    std::sqrt(covariance[i * n + i] * covariance[j * n + j]),
                -1.0, 1.0);
            const double d = std::sqrt(std::max(0.0, 0.5 * (1.0 - corr)));
            distance[i * n + j] = d;
            distance[j * n + i] = d;
        }
    }

    std::vector<std::vector<std::size_t>> members(n);
    std::vector<Node> nodes;
    nodes.reserve(2 * n - 1);
    for (std::size_t i = 0; i < n; ++i) {
        members[i] = {i};
        Node node;
        node.leaf = i;
        node.is_leaf = true;
        nodes.push_back(node);
    }
    std::vector<std::size_t> cluster_node(n);
    std::iota(cluster_node.begin(), cluster_node.end(), 0);
    std::vector<bool> active(n, true);
    std::size_t active_count = n;
    std::size_t root_cluster = 0;

    while (active_count > 1) {
        std::size_t best_a = 0;
        std::size_t best_b = 1;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t a = 0; a < n; ++a) {
            if (!active[a]) continue;
            for (std::size_t b = a + 1; b < n; ++b) {
                if (!active[b]) continue;
                if (distance[a * n + b] < best) {
                    best = distance[a * n + b];
                    best_a = a;
                    best_b = b;
                }
            }
        }
        require(std::isfinite(best), "hierarchical clustering lost connectivity");

        Node merged;
        merged.left = cluster_node[best_a];
        merged.right = cluster_node[best_b];
        nodes.push_back(merged);
        const std::size_t merged_index = nodes.size() - 1;
        members[best_a].insert(members[best_a].end(),
                               members[best_b].begin(), members[best_b].end());
        cluster_node[best_a] = merged_index;
        active[best_b] = false;
        root_cluster = best_a;
        --active_count;

        for (std::size_t other = 0; other < n; ++other) {
            if (!active[other] || other == best_a) continue;
            const double new_distance =
                std::min(distance[best_a * n + other],
                         distance[best_b * n + other]);
            distance[best_a * n + other] = new_distance;
            distance[other * n + best_a] = new_distance;
        }
    }

    const std::size_t root = cluster_node[root_cluster];
    std::vector<std::size_t> order;
    order.reserve(n);
    std::function<void(std::size_t)> visit = [&](std::size_t index) {
        const Node& node = nodes[index];
        if (node.is_leaf) {
            order.push_back(node.leaf);
            return;
        }
        visit(node.left);
        visit(node.right);
    };
    visit(root);
    require(order.size() == n, "hierarchical clustering produced a bad order");
    return order;
}

double cluster_variance(const double* covariance, std::size_t n,
                        const std::vector<std::size_t>& leaves) {
    long double diagonal_sum = 0.0L;
    for (std::size_t leaf : leaves) {
        diagonal_sum += static_cast<long double>(covariance[leaf * n + leaf]);
    }
    if (!(diagonal_sum > 0.0L)) return kNaN;
    std::vector<long double> w(leaves.size(), 0.0L);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        w[i] = 1.0L / static_cast<long double>(covariance[leaves[i] * n + leaves[i]]);
    }
    long double total = std::accumulate(w.begin(), w.end(), 0.0L);
    if (!(total > 0.0L)) return kNaN;
    for (long double& value : w) value /= total;
    long double variance = 0.0L;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        for (std::size_t j = 0; j < leaves.size(); ++j) {
            variance += w[i] * w[j] *
                        static_cast<long double>(
                            covariance[leaves[i] * n + leaves[j]]);
        }
    }
    return static_cast<double>(variance);
}

void allocate(const double* covariance, std::size_t n,
              const std::vector<std::size_t>& order, double budget,
              std::vector<double>& weights, std::size_t begin,
              std::size_t end) {
    if (end - begin == 1) {
        weights[order[begin]] += budget;
        return;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    const std::vector<std::size_t> left(order.begin() + begin,
                                        order.begin() + middle);
    const std::vector<std::size_t> right(order.begin() + middle,
                                         order.begin() + end);
    const double left_var = cluster_variance(covariance, n, left);
    const double right_var = cluster_variance(covariance, n, right);
    double left_budget = budget * 0.5;
    if (std::isfinite(left_var) && std::isfinite(right_var) &&
        left_var + right_var > 0.0) {
        left_budget = budget * (right_var / (left_var + right_var));
    }
    allocate(covariance, n, order, left_budget, weights, begin, middle);
    allocate(covariance, n, order, budget - left_budget, weights, middle, end);
}

}  // namespace

std::vector<std::size_t> quasi_diagonal_order(const double* covariance,
                                              std::size_t n_assets) {
    return cluster_order(covariance, n_assets);
}

std::vector<double> HRPAllocator::weights(const double* covariance,
                                          std::size_t n_assets) const {
    const std::vector<std::size_t> order =
        cluster_order(covariance, n_assets);
    std::vector<double> result(n_assets, 0.0);
    allocate(covariance, n_assets, order, 1.0, result, 0, n_assets);
    double total = 0.0;
    for (double value : result) total += value;
    require(total > 0.0, "HRP produced no allocation");
    for (double& value : result) value /= total;
    return result;
}

}  // namespace rflux::portfoliolab
