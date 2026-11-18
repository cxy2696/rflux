#include "rflux/percolab/network.hpp"

#include "rflux/percolab/percolation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "internal/common.hpp"
#include "internal/graph.hpp"

namespace rflux::percolab {
namespace {

using rflux::detail::checked_mul;
using rflux::detail::component_labels;
using rflux::detail::kNaN;
using rflux::detail::mix_seed;
using rflux::detail::require;
using rflux::detail::require_adjacency;
using rflux::detail::require_finite;
using rflux::detail::require_nonnegative_weights;
using rflux::detail::require_pointer;
using rflux::detail::Rng;
using rflux::detail::UnionFind;
using rflux::detail::validate_symmetric_matrix;

void validate_uint32_nodes(std::size_t n) {
    require(n <= static_cast<std::size_t>(
                     std::numeric_limits<std::uint32_t>::max()),
            "n must fit in uint32_t");
}

/* Node count must fit the int32 labels and distances used by the kernels. */
void validate_indexable_nodes(std::size_t n) {
    validate_uint32_nodes(n);
    require(n <= static_cast<std::size_t>(
                     std::numeric_limits<std::int32_t>::max()),
            "n must fit in int32_t");
}

using NeighborList = std::vector<std::vector<std::size_t>>;

/* Unweighted neighbours (i != j and weights > 0) for one validation pass. */
NeighborList unweighted_neighbors(const double* weights, std::size_t n) {
    NeighborList neighbors(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j && weights[i * n + j] > 0.0) {
                neighbors[i].push_back(j);
            }
        }
    }
    return neighbors;
}

struct ConnectivitySummary {
    std::vector<BridgeEdge> bridges;
    std::vector<std::uint8_t> articulation;
};

/*
 * Iterative Tarjan low-link DFS.  Every non-tree edge in an undirected DFS
 * joins an ancestor and a descendant, so low[v] can be updated from disc[to]
 * for every visited neighbour except the parent.  A tree edge (p, v) is a
 * bridge exactly when low[v] > disc[p]; p is an articulation point when it is
 * not the DFS root and low[v] >= disc[p].  A root is an articulation point
 * exactly when it has more than one DFS child.
 */
ConnectivitySummary analyze_connectivity(const double* weights, std::size_t n) {
    require_adjacency(weights, n, "weights");
    validate_indexable_nodes(n);

    const NeighborList neighbors = unweighted_neighbors(weights, n);
    constexpr std::size_t kUnvisited =
        std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> discovery(n, kUnvisited);
    std::vector<std::size_t> low(n, 0);
    std::vector<std::size_t> parent(n, kUnvisited);
    std::vector<std::size_t> children(n, 0);
    std::vector<std::uint8_t> articulation(n, 0);
    std::vector<BridgeEdge> found;

    struct Frame {
        std::size_t node = 0;
        std::size_t next = 0;
    };
    std::vector<Frame> stack;
    stack.reserve(n);
    std::size_t time = 0;

    for (std::size_t root = 0; root < n; ++root) {
        if (discovery[root] != kUnvisited) continue;
        discovery[root] = low[root] = time++;
        parent[root] = kUnvisited;
        stack.push_back({root, 0});
        while (!stack.empty()) {
            Frame& frame = stack.back();
            if (frame.next < neighbors[frame.node].size()) {
                const std::size_t target =
                    neighbors[frame.node][frame.next++];
                if (target == parent[frame.node]) continue;
                if (discovery[target] == kUnvisited) {
                    parent[target] = frame.node;
                    discovery[target] = low[target] = time++;
                    ++children[frame.node];
                    stack.push_back({target, 0});
                } else {
                    low[frame.node] =
                        std::min(low[frame.node], discovery[target]);
                }
                continue;
            }

            const std::size_t node = frame.node;
            stack.pop_back();
            const std::size_t p = parent[node];
            if (p == kUnvisited) continue;
            low[p] = std::min(low[p], low[node]);
            if (low[node] > discovery[p]) {
                const std::uint32_t left = static_cast<std::uint32_t>(
                    std::min(node, p));
                const std::uint32_t right = static_cast<std::uint32_t>(
                    std::max(node, p));
                found.push_back({left, right});
            }
            if (low[node] >= discovery[p] && parent[p] != kUnvisited) {
                articulation[p] = 1;
            }
        }
        if (children[root] > 1) articulation[root] = 1;
    }

    std::sort(found.begin(), found.end(),
              [](const BridgeEdge& a, const BridgeEdge& b) {
                  if (a.u != b.u) return a.u < b.u;
                  return a.v < b.v;
              });
    ConnectivitySummary summary;
    summary.bridges = std::move(found);
    summary.articulation = std::move(articulation);
    return summary;
}

/*
 * Dense Prim for a complete symmetric distance matrix.
 *
 * The correlation-distance input is always dense, so Kruskal materialised and
 * sorted n(n-1)/2 edges even though only n-1 are used.  Prim keeps O(n) state
 * and does O(n^2) comparisons, which is substantially cheaper for large n and
 * removes the temporary edge array.  Ties select the smallest node first, and
 * the returned edges are sorted by (weight, u, v) so the public output order
 * remains deterministic and compatible with the previous Kruskal order.
 */
std::vector<MstEdge> prim_mst(const double* distance, std::size_t n) {
    if (n <= 1) return {};
    constexpr std::size_t kNone =
        std::numeric_limits<std::size_t>::max();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> best(n, inf);
    std::vector<std::size_t> parent(n, kNone);
    std::vector<std::uint8_t> used(n, 0);
    std::vector<MstEdge> tree;
    tree.reserve(n - 1);
    best[0] = 0.0;

    for (std::size_t iteration = 0; iteration < n; ++iteration) {
        std::size_t node = kNone;
        double key = inf;
        for (std::size_t candidate = 0; candidate < n; ++candidate) {
            if (used[candidate]) continue;
            if (best[candidate] < key ||
                (best[candidate] == key &&
                 (node == kNone || candidate < node))) {
                node = candidate;
                key = best[candidate];
            }
        }
        if (node == kNone || !std::isfinite(key)) {
            throw NumericalError("MST could not connect all nodes");
        }
        used[node] = 1;
        if (parent[node] != kNone) {
            const std::uint32_t left = static_cast<std::uint32_t>(
                std::min(node, parent[node]));
            const std::uint32_t right = static_cast<std::uint32_t>(
                std::max(node, parent[node]));
            tree.push_back({left, right, key});
        }
        for (std::size_t other = 0; other < n; ++other) {
            if (used[other]) continue;
            const double weight = distance[node * n + other];
            if (weight < best[other]) {
                best[other] = weight;
                parent[other] = node;
            }
        }
    }

    std::sort(tree.begin(), tree.end(),
              [](const MstEdge& a, const MstEdge& b) {
                  if (a.weight != b.weight) return a.weight < b.weight;
                  if (a.u != b.u) return a.u < b.u;
                  return a.v < b.v;
              });
    return tree;
}

void validate_diffusion_inputs(const double* adjacency, std::size_t n,
                               double beta, double gamma, std::size_t steps,
                               std::size_t repetitions) {
    require_adjacency(adjacency, n, "adjacency");
    require(std::isfinite(beta) && beta >= 0.0,
            "beta must be finite and non-negative");
    rflux::detail::require_probability(gamma, "gamma");
    require(steps > 0, "steps must be positive");
    require(steps < std::numeric_limits<std::size_t>::max(),
            "steps is too large");
    require(repetitions > 0, "repetitions must be positive");
    validate_uint32_nodes(n);
}

}  // namespace

std::vector<double> correlation_to_distance(const double* corr, std::size_t n) {
    validate_symmetric_matrix(corr, n, "corr");
    std::vector<double> out(checked_mul(n, n, "n*n overflows size_t"), kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        out[i * n + i] = 0.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            const double value = corr[i * n + j];
            if (value < -1.0 - 1e-9 || value > 1.0 + 1e-9) {
                throw InvalidArgument(
                    "correlation entries must be in [-1, 1]");
            }
            const double clipped = std::max(-1.0, std::min(1.0, value));
            const double distance =
                std::sqrt(std::max(0.0, 2.0 * (1.0 - clipped)));
            out[i * n + j] = distance;
            out[j * n + i] = distance;
        }
    }
    return out;
}


CorrelationAdjacency correlation_adjacency(const double* correlation,
                                           std::size_t n, double threshold,
                                           bool weighted) {
    rflux::detail::require_pointer(correlation,
                                   "correlation must not be NULL");
    rflux::detail::require(n > 0, "n must be positive");
    rflux::detail::checked_mul(n, n, "matrix dimensions overflow");
    rflux::detail::require(std::isfinite(threshold) && threshold > 0.0 &&
                               threshold <= 1.0,
                           "threshold must be in (0, 1]");
    for (std::size_t i = 0; i < n * n; ++i) {
        if (!std::isfinite(correlation[i])) {
            throw rflux::NonFinite(
                "correlation matrix contains a non-finite value");
        }
    }
    CorrelationAdjacency result;
    result.adjacency.assign(n * n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            if (row == column) continue;
            const double value = correlation[row * n + column];
            if (value >= threshold) {
                result.adjacency[row * n + column] =
                    weighted ? std::max(value, 0.0) : 1.0;
            }
        }
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = row + 1; column < n; ++column) {
            const double value =
                0.5 * (result.adjacency[row * n + column] +
                       result.adjacency[column * n + row]);
            result.adjacency[row * n + column] = value;
            result.adjacency[column * n + row] = value;
        }
        result.adjacency[row * n + row] = 0.0;
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = row + 1; column < n; ++column) {
            if (result.adjacency[row * n + column] > 0.0) {
                ++result.n_edges;
            }
        }
    }
    result.density =
        n > 1 ? 2.0 * static_cast<double>(result.n_edges) /
                    (static_cast<double>(n) * static_cast<double>(n - 1))
              : 0.0;
    return result;
}

std::vector<std::size_t> component_sizes(const std::int32_t* labels,
                                         std::size_t n,
                                         std::size_t n_components) {
    rflux::detail::require_pointer(labels, "labels must not be NULL");
    rflux::detail::require(n > 0, "n must be positive");
    rflux::detail::require(n_components > 0,
                           "n_components must be positive");
    std::vector<std::size_t> sizes(n_components, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (labels[i] < 0 ||
            static_cast<std::size_t>(labels[i]) >= n_components) {
            throw rflux::InvalidArgument(
                "labels must be in the component range");
        }
        ++sizes[static_cast<std::size_t>(labels[i])];
    }
    return sizes;
}

std::vector<double> strength(const double* weights, std::size_t n) {
    require_adjacency(weights, n, "weights");
    std::vector<double> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        long double sum = 0.0L;
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            sum += static_cast<long double>(weights[i * n + j]);
        }
        out[i] = static_cast<double>(sum);
    }
    return out;
}

std::vector<double> eigenvector_centrality(const double* weights,
                                           std::size_t n) {
    require_adjacency(weights, n, "weights");

    bool has_edge = false;
    for (std::size_t i = 0; i < n && !has_edge; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (weights[i * n + j] > 0.0) {
                has_edge = true;
                break;
            }
        }
    }
    if (!has_edge) return std::vector<double>(n, 0.0);

    // Power iteration on A + I, not A.  For a bipartite graph the spectrum of
    // A is symmetric around zero, so plain power iteration alternates between
    // two vectors and returns a non-eigenvector at the iteration cap.  Adding
    // the identity shifts every eigenvalue by one without changing the
    // eigenvectors, making the largest eigenvalue strictly dominant.
    std::vector<double> current(n, 1.0 / std::sqrt(static_cast<double>(n)));
    std::vector<double> next(n, 0.0);
    constexpr std::size_t kMaxIterations = 2000;
    constexpr double kTolerance = 1e-10;
    bool converged = false;

    for (std::size_t iteration = 0; iteration < kMaxIterations; ++iteration) {
        long double norm_squared = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            long double sum = static_cast<long double>(current[i]);
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                sum += static_cast<long double>(weights[i * n + j]) *
                       static_cast<long double>(current[j]);
            }
            next[i] = static_cast<double>(sum);
            norm_squared += static_cast<long double>(next[i]) *
                            static_cast<long double>(next[i]);
        }
        if (!(norm_squared > 0.0L)) {
            throw NumericalError("eigenvector centrality lost all mass");
        }
        const double norm = std::sqrt(static_cast<double>(norm_squared));
        double difference = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            next[i] /= norm;
            difference = std::max(difference, std::abs(next[i] - current[i]));
        }
        current.swap(next);
        if (difference < kTolerance) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        throw NumericalError("eigenvector centrality did not converge");
    }
    return current;
}

std::vector<double> clustering_coefficients(const double* weights,
                                            std::size_t n) {
    require_adjacency(weights, n, "weights");
    std::vector<double> out(n, kNaN);
    const NeighborList neighbors = unweighted_neighbors(weights, n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t degree = neighbors[i].size();
        if (degree < 2) continue;
        std::size_t links = 0;
        for (std::size_t a = 0; a < degree; ++a) {
            for (std::size_t b = a + 1; b < degree; ++b) {
                if (weights[neighbors[i][a] * n + neighbors[i][b]] > 0.0) {
                    ++links;
                }
            }
        }
        out[i] = 2.0 * static_cast<double>(links) /
                 (static_cast<double>(degree) *
                  static_cast<double>(degree - 1));
    }
    return out;
}

double transitivity(const double* weights, std::size_t n) {
    require_adjacency(weights, n, "weights");
    const NeighborList neighbors = unweighted_neighbors(weights, n);

    long double triples = 0.0L;
    for (const std::vector<std::size_t>& list : neighbors) {
        const long double degree = static_cast<long double>(list.size());
        triples += degree * (degree - 1.0L) / 2.0L;
    }
    if (!(triples > 0.0L)) return kNaN;

    // Sum over edges the number of common neighbours.  Each triangle is
    // discovered once per edge, so the sum equals three times the triangle
    // count and the transitivity is sum / triples.
    long double edge_common_neighbors = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j : neighbors[i]) {
            if (j <= i) continue;
            std::size_t common = 0;
            for (std::size_t k : neighbors[i]) {
                if (k != j && weights[j * n + k] > 0.0) ++common;
            }
            edge_common_neighbors += static_cast<long double>(common);
        }
    }
    return static_cast<double>(edge_common_neighbors / triples);
}

PathStats path_statistics(const double* weights, std::size_t n) {
    require_adjacency(weights, n, "weights");
    validate_indexable_nodes(n);
    const NeighborList neighbors = unweighted_neighbors(weights, n);

    PathStats result;
    result.connected = true;
    std::vector<std::int32_t> distance(n, -1);
    std::vector<std::size_t> queue(n, 0);
    long double distance_sum = 0.0L;
    long double inverse_sum = 0.0L;
    std::size_t reachable = 0;
    std::size_t diameter = 0;

    for (std::size_t source = 0; source < n; ++source) {
        std::fill(distance.begin(), distance.end(), -1);
        std::size_t head = 0;
        std::size_t tail = 0;
        distance[source] = 0;
        queue[tail++] = source;
        while (head < tail) {
            const std::size_t node = queue[head++];
            for (std::size_t next : neighbors[node]) {
                if (distance[next] >= 0) continue;
                distance[next] = distance[node] + 1;
                queue[tail++] = next;
            }
        }
        for (std::size_t target = source + 1; target < n; ++target) {
            if (distance[target] < 0) {
                result.connected = false;
                continue;
            }
            const std::size_t value =
                static_cast<std::size_t>(distance[target]);
            ++reachable;
            distance_sum += static_cast<long double>(value);
            inverse_sum += 1.0L / static_cast<long double>(value);
            if (value > diameter) diameter = value;
        }
    }

    result.n_reachable_pairs = reachable;
    result.diameter = diameter;
    result.average_path_length =
        reachable > 0
            ? static_cast<double>(distance_sum /
                                  static_cast<long double>(reachable))
            : kNaN;
    if (n > 1) {
        const long double ordered_pairs =
            static_cast<long double>(n) * static_cast<long double>(n - 1);
        result.efficiency =
            static_cast<double>(2.0L * inverse_sum / ordered_pairs);
    } else {
        result.efficiency = 0.0;
    }
    return result;
}


std::vector<std::uint32_t> removal_order(const double* adjacency,
                                         std::size_t n,
                                         RemovalStrategy strategy) {
    rflux::detail::require_pointer(adjacency, "adjacency must not be NULL");
    rflux::detail::require(n > 0, "n must be positive");
    rflux::detail::checked_mul(n, n, "matrix dimensions overflow");
    std::vector<double> scores(n, 0.0);
    switch (strategy) {
        case RemovalStrategy::strength:
            scores = strength(adjacency, n);
            break;
        case RemovalStrategy::degree: {
            for (std::size_t row = 0; row < n; ++row) {
                std::size_t degree = 0;
                for (std::size_t column = 0; column < n; ++column) {
                    if (row != column && adjacency[row * n + column] > 0.0) {
                        ++degree;
                    }
                }
                scores[row] = static_cast<double>(degree);
            }
            break;
        }
        case RemovalStrategy::eigenvector:
            scores = eigenvector_centrality(adjacency, n);
            break;
        case RemovalStrategy::betweenness:
            scores = betweenness_centrality(adjacency, n, false);
            break;
        case RemovalStrategy::core: {
            const std::vector<std::int32_t> core = core_numbers(adjacency, n);
            for (std::size_t i = 0; i < n; ++i) {
                scores[i] = -static_cast<double>(core[i]);
            }
            break;
        }
    }
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t left, std::size_t right) {
                         return scores[left] > scores[right];
                     });
    std::vector<std::uint32_t> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = static_cast<std::uint32_t>(order[i]);
    }
    return result;
}

CoreCurve core_percolation(const double* adjacency, std::size_t n) {
    rflux::detail::require_pointer(adjacency, "adjacency must not be NULL");
    rflux::detail::require(n > 0, "n must be positive");
    const std::vector<std::int32_t> core = core_numbers(adjacency, n);
    std::int32_t maximum = 0;
    for (std::int32_t value : core) maximum = std::max(maximum, value);
    CoreCurve result;
    for (std::int32_t k = 1; k <= maximum; ++k) {
        std::int64_t count = 0;
        for (std::int32_t value : core) {
            if (value >= k) ++count;
        }
        result.k.push_back(static_cast<std::int64_t>(k));
        result.n_nodes.push_back(count);
        result.fraction.push_back(
            static_cast<double>(count) / static_cast<double>(n));
    }
    return result;
}

std::vector<std::int32_t> core_numbers(const double* weights, std::size_t n) {
    require_adjacency(weights, n, "weights");
    validate_indexable_nodes(n);

    std::vector<std::int32_t> degree(n, 0);
    std::int32_t max_degree = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t count = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j && weights[i * n + j] > 0.0) ++count;
        }
        degree[i] = static_cast<std::int32_t>(count);
        max_degree = std::max(max_degree, degree[i]);
    }

    // Batagelj-Zaversnik degree buckets.  Vertices are processed in
    // non-decreasing current degree; a neighbour is moved to the front of its
    // new bucket only when its degree is strictly larger than the current
    // vertex's degree.
    std::vector<std::int32_t> bin(static_cast<std::size_t>(max_degree) + 1,
                                  0);
    for (std::size_t v = 0; v < n; ++v) {
        ++bin[static_cast<std::size_t>(degree[v])];
    }
    std::int32_t start = 0;
    for (std::int32_t d = 0; d <= max_degree; ++d) {
        const std::int32_t count = bin[static_cast<std::size_t>(d)];
        bin[static_cast<std::size_t>(d)] = start;
        start += count;
    }
    std::vector<std::int32_t> vert(n, 0);
    std::vector<std::int32_t> pos(n, 0);
    for (std::size_t v = 0; v < n; ++v) {
        const std::size_t d = static_cast<std::size_t>(degree[v]);
        pos[v] = bin[d];
        vert[static_cast<std::size_t>(bin[d])] =
            static_cast<std::int32_t>(v);
        ++bin[d];
    }
    for (std::int32_t d = max_degree; d >= 1; --d) {
        bin[static_cast<std::size_t>(d)] =
            bin[static_cast<std::size_t>(d - 1)];
    }
    bin[0] = 0;

    std::vector<std::int32_t> core(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t v = vert[i];
        const std::size_t node = static_cast<std::size_t>(v);
        core[node] = degree[node];
        for (std::size_t u = 0; u < n; ++u) {
            if (u == node || weights[node * n + u] <= 0.0) continue;
            if (degree[u] > degree[node]) {
                const std::int32_t old_degree = degree[u];
                const std::int32_t old_position = pos[u];
                const std::int32_t bucket =
                    bin[static_cast<std::size_t>(old_degree)];
                const std::int32_t displaced =
                    vert[static_cast<std::size_t>(bucket)];
                if (u != static_cast<std::size_t>(displaced)) {
                    pos[u] = bucket;
                    vert[static_cast<std::size_t>(old_position)] = displaced;
                    pos[static_cast<std::size_t>(displaced)] = old_position;
                    vert[static_cast<std::size_t>(bucket)] =
                        static_cast<std::int32_t>(u);
                }
                ++bin[static_cast<std::size_t>(old_degree)];
                --degree[u];
            }
        }
    }
    return core;
}

double degree_assortativity(const double* weights, std::size_t n) {
    require_adjacency(weights, n, "weights");
    std::vector<long double> degree(n, 0.0L);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (weights[i * n + j] > 0.0) {
                degree[i] += 1.0L;
                degree[j] += 1.0L;
            }
        }
    }

    long double edges = 0.0L;
    long double product_sum = 0.0L;
    long double half_sum = 0.0L;
    long double half_square_sum = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (weights[i * n + j] <= 0.0) continue;
            const long double a = degree[i];
            const long double b = degree[j];
            edges += 1.0L;
            product_sum += a * b;
            half_sum += (a + b) / 2.0L;
            half_square_sum += (a * a + b * b) / 2.0L;
        }
    }
    if (!(edges > 0.0L)) return kNaN;

    const long double mean_product = product_sum / edges;
    const long double mean_half = half_sum / edges;
    const long double mean_half_square = half_square_sum / edges;
    const long double denominator =
        mean_half_square - mean_half * mean_half;
    const long double scale = std::max(1.0L, std::fabs(mean_half_square));
    if (denominator <= 1e-18L * scale) return kNaN;
    return static_cast<double>((mean_product - mean_half * mean_half) /
                               denominator);
}

double modularity(const double* weights, std::size_t n,
                  const std::int32_t* labels, std::size_t n_labels) {
    require_adjacency(weights, n, "weights");
    require_pointer(labels, "labels must not be NULL");
    require(n_labels > 0, "n_labels must be positive");
    require(n_labels <= n, "n_labels must not exceed n");
    for (std::size_t i = 0; i < n; ++i) {
        require(labels[i] >= 0 &&
                    static_cast<std::size_t>(labels[i]) < n_labels,
                "labels must be in [0, n_labels)");
    }

    std::vector<long double> degree(n, 0.0L);
    long double two_m = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const long double value =
                static_cast<long double>(weights[i * n + j]);
            degree[i] += value;
            two_m += value;
        }
    }
    if (!(two_m > 0.0L)) return kNaN;

    std::vector<long double> community_strength(n_labels, 0.0L);
    for (std::size_t i = 0; i < n; ++i) {
        community_strength[static_cast<std::size_t>(labels[i])] += degree[i];
    }

    std::vector<long double> internal_weight(n_labels, 0.0L);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (labels[i] != labels[j]) continue;
            const double value = weights[i * n + j];
            if (value <= 0.0) continue;
            internal_weight[static_cast<std::size_t>(labels[i])] +=
                2.0L * static_cast<long double>(value);
        }
    }

    long double result = 0.0L;
    for (std::size_t c = 0; c < n_labels; ++c) {
        const long double share = community_strength[c] / two_m;
        result += internal_weight[c] / two_m - share * share;
    }
    return static_cast<double>(result);
}

std::vector<double> pagerank(const double* adjacency, std::size_t n,
                             double damping, std::size_t max_iter, double tol) {
    require_pointer(adjacency, "adjacency must not be NULL");
    require(n > 0, "n must be positive");
    require_finite(adjacency, checked_mul(n, n, "n*n overflows size_t"),
                   "adjacency");
    require_nonnegative_weights(adjacency, n, "adjacency");
    require(std::isfinite(damping) && damping >= 0.0 && damping < 1.0,
            "damping must be in [0, 1)");
    require(max_iter > 0, "max_iter must be positive");
    require(std::isfinite(tol) && tol > 0.0,
            "tol must be finite and positive");

    std::vector<long double> row_sum(n, 0.0L);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            row_sum[i] += static_cast<long double>(adjacency[i * n + j]);
        }
    }

    std::vector<double> rank(n, 1.0 / static_cast<double>(n));
    std::vector<double> next(n, 0.0);
    bool converged = false;
    const long double teleport =
        (1.0L - static_cast<long double>(damping)) /
        static_cast<long double>(n);
    for (std::size_t iteration = 0; iteration < max_iter; ++iteration) {
        long double dangling = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            if (row_sum[i] == 0.0L) {
                dangling += static_cast<long double>(rank[i]);
            }
        }
        for (std::size_t j = 0; j < n; ++j) {
            long double incoming = 0.0L;
            for (std::size_t i = 0; i < n; ++i) {
                if (row_sum[i] > 0.0L) {
                    incoming +=
                        static_cast<long double>(rank[i]) *
                        static_cast<long double>(adjacency[i * n + j]) /
                        row_sum[i];
                }
            }
            next[j] = static_cast<double>(
                teleport +
                static_cast<long double>(damping) *
                    (incoming + dangling / static_cast<long double>(n)));
        }
        long double difference = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            difference += std::abs(static_cast<long double>(next[i]) -
                                   static_cast<long double>(rank[i]));
        }
        rank.swap(next);
        if (difference <= static_cast<long double>(tol)) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        throw NumericalError("PageRank did not converge");
    }
    long double total = 0.0L;
    for (double value : rank) total += static_cast<long double>(value);
    if (!(total > 0.0L)) {
        throw NumericalError("PageRank lost probability mass");
    }
    for (double& value : rank) {
        value = static_cast<double>(static_cast<long double>(value) / total);
    }
    return rank;
}

std::vector<double> betweenness_centrality(const double* adjacency,
                                           std::size_t n, bool normalized) {
    require_adjacency(adjacency, n, "adjacency");
    const NeighborList neighbors = unweighted_neighbors(adjacency, n);

    std::vector<long double> raw(n, 0.0L);
    std::vector<int> distance(n, -1);
    std::vector<long double> paths(n, 0.0L);
    std::vector<long double> delta(n, 0.0L);
    std::vector<std::vector<std::size_t>> predecessors(n);
    std::vector<std::size_t> queue(n, 0);
    std::vector<std::size_t> order;
    order.reserve(n);

    for (std::size_t source = 0; source < n; ++source) {
        std::fill(distance.begin(), distance.end(), -1);
        std::fill(paths.begin(), paths.end(), 0.0L);
        std::fill(delta.begin(), delta.end(), 0.0L);
        for (auto& list : predecessors) list.clear();
        order.clear();

        distance[source] = 0;
        paths[source] = 1.0L;
        std::size_t head = 0;
        std::size_t tail = 0;
        queue[tail++] = source;
        while (head < tail) {
            const std::size_t node = queue[head++];
            order.push_back(node);
            for (std::size_t next : neighbors[node]) {
                if (distance[next] < 0) {
                    distance[next] = distance[node] + 1;
                    queue[tail++] = next;
                }
                if (distance[next] == distance[node] + 1) {
                    paths[next] += paths[node];
                    predecessors[next].push_back(node);
                }
            }
        }

        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const std::size_t node = *it;
            for (std::size_t previous : predecessors[node]) {
                delta[previous] +=
                    paths[previous] / paths[node] * (1.0L + delta[node]);
            }
            if (node != source) raw[node] += delta[node];
        }
    }

    const long double scale =
        normalized && n > 2
            ? static_cast<long double>((n - 1) * (n - 2))
            : 2.0L;
    std::vector<double> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<double>(raw[i] / scale);
    }
    return out;
}

std::vector<BridgeEdge> bridges(const double* adjacency, std::size_t n) {
    return analyze_connectivity(adjacency, n).bridges;
}

std::vector<std::int32_t> articulation_points(const double* adjacency,
                                              std::size_t n) {
    const ConnectivitySummary summary = analyze_connectivity(adjacency, n);
    std::vector<std::int32_t> out(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = summary.articulation[i] != 0 ? 1 : 0;
    }
    return out;
}

std::vector<std::int32_t> truss_numbers(const double* adjacency,
                                        std::size_t n) {
    require_adjacency(adjacency, n, "adjacency");
    validate_indexable_nodes(n);
    const std::size_t cells = checked_mul(n, n, "n*n overflows size_t");
    constexpr std::size_t kNoEdge = std::numeric_limits<std::size_t>::max();

    struct Edge {
        std::size_t u;
        std::size_t v;
    };
    std::vector<Edge> edges;
    std::vector<std::size_t> edge_index(cells, kNoEdge);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (adjacency[i * n + j] <= 0.0) continue;
            const std::size_t id = edges.size();
            edges.push_back({i, j});
            edge_index[i * n + j] = id;
            edge_index[j * n + i] = id;
        }
    }

    const std::size_t m = edges.size();
    std::vector<std::size_t> support(m, 0);
    std::size_t max_support = 0;
    for (std::size_t e = 0; e < m; ++e) {
        const std::size_t u = edges[e].u;
        const std::size_t v = edges[e].v;
        std::size_t count = 0;
        for (std::size_t w = 0; w < n; ++w) {
            if (w == u || w == v) continue;
            if (adjacency[u * n + w] > 0.0 &&
                adjacency[v * n + w] > 0.0) {
                ++count;
            }
        }
        support[e] = count;
        max_support = std::max(max_support, count);
    }

    using Entry = std::pair<std::size_t, std::size_t>; /* support, edge */
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>
        queue;
    for (std::size_t e = 0; e < m; ++e) {
        queue.push({support[e], e});
    }
    std::vector<std::uint8_t> alive(m, 1);
    std::vector<std::int32_t> trussness(m, 0);

    /*
     * Stage k removes every edge whose support is below k - 2.  An edge
     * removed in stage k survived stages 3..k-1, so its trussness is k - 1.
     * Supports only decrease, so a lazy min-heap with the current support is
     * enough; every triangle loss pushes the two remaining edges again.
     */
    const std::size_t max_k = max_support + 2;
    for (std::size_t k = 3; k <= max_k; ++k) {
        const std::size_t threshold = k - 2;
        while (!queue.empty() && queue.top().first < threshold) {
            const Entry entry = queue.top();
            queue.pop();
            const std::size_t e = entry.second;
            if (alive[e] == 0 || support[e] != entry.first) continue;
            alive[e] = 0;
            trussness[e] = static_cast<std::int32_t>(k - 1);

            const std::size_t u = edges[e].u;
            const std::size_t v = edges[e].v;
            for (std::size_t w = 0; w < n; ++w) {
                if (w == u || w == v) continue;
                if (adjacency[u * n + w] <= 0.0 ||
                    adjacency[v * n + w] <= 0.0) {
                    continue;
                }
                const std::size_t first = edge_index[u * n + w];
                const std::size_t second = edge_index[v * n + w];
                if (first == kNoEdge || second == kNoEdge) continue;
                if (alive[first] == 0 || alive[second] == 0) continue;
                --support[first];
                queue.push({support[first], first});
                --support[second];
                queue.push({support[second], second});
            }
        }
    }

    for (std::size_t e = 0; e < m; ++e) {
        if (alive[e] != 0) {
            if (support[e] + 2 < max_k) {
                throw NumericalError("truss peeling invariant violated");
            }
            trussness[e] = static_cast<std::int32_t>(max_k);
        }
    }

    std::vector<std::int32_t> out(cells, 0);
    for (std::size_t e = 0; e < m; ++e) {
        const std::size_t u = edges[e].u;
        const std::size_t v = edges[e].v;
        out[u * n + v] = trussness[e];
        out[v * n + u] = trussness[e];
    }
    return out;
}

std::vector<MstEdge> minimum_spanning_tree(const double* distance,
                                           std::size_t n) {
    require_adjacency(distance, n, "distance");
    validate_uint32_nodes(n);
    return prim_mst(distance, n);
}

ComponentSummary mst_clusters(const double* distance, std::size_t n,
                              double threshold) {
    require_adjacency(distance, n, "distance");
    validate_uint32_nodes(n);
    require(std::isfinite(threshold), "threshold must be finite");
    const std::vector<MstEdge> tree = prim_mst(distance, n);
    UnionFind uf(n);
    for (const MstEdge& edge : tree) {
        if (edge.weight <= threshold) uf.unite(edge.u, edge.v);
    }
    ComponentSummary summary;
    summary.labels =
        component_labels(uf, n, &summary.n_components, &summary.largest_size);
    return summary;
}

DiffusionResult risk_diffusion(const double* adjacency, std::size_t n,
                               const std::uint32_t* seed_nodes,
                               std::size_t n_seed_nodes, double beta,
                               double gamma, std::size_t steps,
                               std::size_t repetitions, std::uint64_t seed) {
    validate_diffusion_inputs(adjacency, n, beta, gamma, steps, repetitions);
    if (n_seed_nodes > 0) {
        require_pointer(seed_nodes, "seed_nodes must not be NULL");
    }
    std::vector<std::uint8_t> is_seed(n, 0);
    std::size_t initial_count = 0;
    for (std::size_t k = 0; k < n_seed_nodes; ++k) {
        require(static_cast<std::size_t>(seed_nodes[k]) < n,
                "seed node index is out of range");
        const std::size_t node = static_cast<std::size_t>(seed_nodes[k]);
        if (is_seed[node] == 0) {
            is_seed[node] = 1;
            ++initial_count;
        }
    }

    DiffusionResult result;
    result.mean_affected.assign(steps + 1, 0.0);
    result.infection_frequency.assign(n, 0.0);
    std::vector<long double> affected_sum(steps + 1, 0.0L);
    std::vector<long double> frequency_sum(n, 0.0L);

    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        Rng rng(mix_seed(seed, repetition));
        std::vector<std::uint8_t> state(n, 0); /* 0=S, 1=I, 2=R */
        std::vector<std::uint8_t> ever(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            if (is_seed[i] != 0) {
                state[i] = 1;
                ever[i] = 1;
            }
        }
        std::size_t affected = initial_count;
        affected_sum[0] +=
            static_cast<long double>(affected) / static_cast<long double>(n);

        std::vector<std::size_t> infected;
        std::vector<std::size_t> newly_infected;
        std::vector<std::size_t> recovered;
        std::vector<std::uint8_t> marked(n, 0);
        for (std::size_t step = 1; step <= steps; ++step) {
            infected.clear();
            for (std::size_t i = 0; i < n; ++i) {
                if (state[i] == 1) infected.push_back(i);
            }
            newly_infected.clear();
            recovered.clear();
            std::fill(marked.begin(), marked.end(), 0);

            for (std::size_t source : infected) {
                for (std::size_t target = 0; target < n; ++target) {
                    if (target == source || state[target] != 0) continue;
                    const double weight = adjacency[source * n + target];
                    if (!(weight > 0.0)) continue;
                    const double probability =
                        1.0 - std::exp(-beta * weight);
                    if (rng.uniform() < probability && marked[target] == 0) {
                        marked[target] = 1;
                        newly_infected.push_back(target);
                    }
                }
            }
            for (std::size_t source : infected) {
                if (rng.uniform() < gamma) recovered.push_back(source);
            }
            for (std::size_t node : recovered) state[node] = 2;
            for (std::size_t node : newly_infected) {
                if (state[node] == 0) {
                    state[node] = 1;
                    ever[node] = 1;
                }
            }
            affected = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (state[i] != 0) ++affected;
            }
            affected_sum[step] +=
                static_cast<long double>(affected) /
                static_cast<long double>(n);
        }
        for (std::size_t i = 0; i < n; ++i) {
            frequency_sum[i] += static_cast<long double>(ever[i]);
        }
    }

    const long double reps = static_cast<long double>(repetitions);
    for (std::size_t step = 0; step <= steps; ++step) {
        result.mean_affected[step] =
            static_cast<double>(affected_sum[step] / reps);
    }
    for (std::size_t i = 0; i < n; ++i) {
        result.infection_frequency[i] =
            static_cast<double>(frequency_sum[i] / reps);
    }
    return result;
}

}  // namespace rflux::percolab
