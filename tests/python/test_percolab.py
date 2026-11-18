import unittest

import numpy as np

from rflux import percolab
from rflux._ffi import RFluxError


CORR = np.array(
    [
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.3, 0.1],
        [0.2, 0.3, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0],
    ]
)

COMPLETE = np.ones((4, 4)) - np.eye(4)
PATH4 = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

CYCLE4 = np.array(
    [
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ]
)


def random_graph(rng: np.random.Generator, n: int = 6,
                 probability: float = 0.4) -> np.ndarray:
    """Undirected unweighted graph with the given edge probability."""
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < probability:
                matrix[i, j] = matrix[j, i] = 1.0
    return matrix


class TestGraphs(unittest.TestCase):
    def test_distance_components_mst(self):
        distance = percolab.correlation_to_distance(CORR)
        self.assertAlmostEqual(distance[0, 1], np.sqrt(0.2))
        np.testing.assert_allclose(distance, distance.T)
        components = percolab.threshold_components(CORR, 0.5)
        self.assertEqual(components.n_components, 2)
        self.assertEqual(components.largest_size, 2)
        np.testing.assert_array_equal(components.sizes, [2, 2])
        self.assertAlmostEqual(components.giant_fraction, 0.5)

        tree = percolab.mst_from_correlation(CORR)
        self.assertEqual(tree.weight.size, 3)
        self.assertTrue(np.all(tree.u != tree.v))
        self.assertTrue(np.all(tree.weight >= 0.0))
        matrix = tree.as_array()
        self.assertEqual(matrix.shape, (3, 3))

    def test_mst_dense_prim_weight_and_order(self):
        distance = np.array(
            [
                [0.0, 1.0, 4.0, 3.0],
                [1.0, 0.0, 2.0, 5.0],
                [4.0, 2.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ]
        )
        tree = percolab.mst_from_distance(distance)
        np.testing.assert_array_equal(tree.u, [0, 1, 0])
        np.testing.assert_array_equal(tree.v, [1, 2, 3])
        np.testing.assert_allclose(tree.weight, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(tree.weight.sum()), 6.0)

    def test_centrality_clustering_and_network(self):
        strength = percolab.network_strength(CORR)
        np.testing.assert_allclose(strength, [1.2, 1.3, 1.3, 1.0])
        eigen = percolab.centrality(CORR, "eigenvector")
        self.assertAlmostEqual(float(np.sum(eigen**2)), 1.0)
        path = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        path_eigen = percolab.centrality(path, "eigenvector")
        np.testing.assert_allclose(
            path_eigen, [0.5, np.sqrt(0.5), 0.5], atol=1e-8
        )
        star = np.zeros((4, 4))
        star[0, 1:] = 1.0
        star[1:, 0] = 1.0
        star_eigen = percolab.centrality(star, "eigenvector")
        self.assertAlmostEqual(float(star_eigen[0]), np.sqrt(0.5), places=8)
        np.testing.assert_allclose(star_eigen[1:], 0.40824829046, atol=1e-8)

        triangle = np.full((3, 3), 0.8)
        np.fill_diagonal(triangle, 1.0)
        clustering = percolab.clustering(triangle)
        np.testing.assert_allclose(clustering, 1.0)
        network = percolab.network_from_correlation(triangle, 0.5)
        self.assertEqual(network.n_edges, 3)
        self.assertAlmostEqual(network.density, 1.0)
        self.assertAlmostEqual(network.average_clustering, 1.0)
        self.assertEqual(network.mst.weight.size, 2)

        with self.assertRaises(ValueError):
            percolab.centrality(CORR, "pagerank")
        with self.assertRaises(RFluxError):
            percolab.threshold_components(
                np.array([[1.0, 0.5], [0.1, 1.0]]), 0.2
            )

    def test_betweenness_on_path_and_star(self):
        path = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        np.testing.assert_allclose(
            percolab.betweenness(path), [0.0, 1.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            percolab.betweenness(path, normalized=False),
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )
        star = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(
            percolab.betweenness(star), [1.0, 0.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            percolab.betweenness(star, normalized=False),
            [3.0, 0.0, 0.0, 0.0],
            atol=1e-12,
        )
        with self.assertRaises(Exception):
            percolab.betweenness(np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_pagerank_directed_and_dangling(self):
        cycle = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(
            percolab.pagerank(cycle), [1.0 / 3.0] * 3, atol=1e-10
        )
        chain = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )
        ranks = percolab.pagerank(chain)
        self.assertAlmostEqual(float(ranks.sum()), 1.0, places=12)
        self.assertGreater(ranks[2], ranks[1])
        self.assertGreater(ranks[1], ranks[0])
        with self.assertRaises(RFluxError):
            percolab.pagerank(cycle, damping=1.0)
        with self.assertRaises(RFluxError):
            percolab.pagerank(np.array([[-1.0, 0.0], [0.0, 0.0]]))

    def test_mst_single_node(self):
        tree = percolab.mst_from_distance(np.array([[0.0]]))
        self.assertEqual(tree.u.size, 0)
        self.assertEqual(tree.as_array().shape, (0, 3))

    def test_percolation_endpoints(self):
        probabilities = np.array([0.0, 1.0])
        bond = percolab.bond_percolation(
            COMPLETE, probabilities, repetitions=20, seed=3
        )
        np.testing.assert_allclose(bond.giant_mean, [0.25, 1.0])
        self.assertAlmostEqual(bond.giant_std[0], 0.0, places=8)
        np.testing.assert_allclose(bond.susceptibility, [0.75, 0.0])
        site = percolab.site_percolation(
            COMPLETE, probabilities, repetitions=20, seed=3
        )
        np.testing.assert_allclose(site.giant_mean, [0.0, 1.0])

    def test_weighted_percolation_endpoints(self):
        complete = np.ones((4, 4)) - np.eye(4)
        weighted_bond = percolab.bond_percolation_weighted(complete, [0.0, 1.0])
        np.testing.assert_allclose(weighted_bond.giant_mean, [0.25, 1.0])
        weighted_site = percolab.site_percolation_weighted(
            complete, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0]
        )
        np.testing.assert_allclose(weighted_site.giant_mean, [0.0, 0.25])
        with self.assertRaises(ValueError):
            percolab.site_percolation_weighted(complete, [1.0], [1.0])

    def test_risk_diffusion_rejects_impossible_step_count(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaises(ValueError):
            percolab.risk_diffusion(
                adjacency, [0], steps=int(np.iinfo(np.uintp).max)
            )

    def test_risk_diffusion(self):
        result = percolab.risk_diffusion(
            COMPLETE,
            [0],
            beta=10.0,
            gamma=0.0,
            steps=3,
            repetitions=10,
            seed=9,
        )
        self.assertEqual(result.mean_affected.shape, (4,))
        self.assertAlmostEqual(result.mean_affected[0], 0.25)
        self.assertGreater(result.mean_affected[1], 0.9)
        self.assertAlmostEqual(result.infection_frequency[0], 1.0)
        self.assertAlmostEqual(result.final_affected(), result.mean_affected[-1])
        for value in result.infection_frequency:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        empty = percolab.risk_diffusion(
            COMPLETE, [], beta=1.0, gamma=0.0, steps=2, repetitions=3
        )
        np.testing.assert_allclose(empty.mean_affected, 0.0)
        np.testing.assert_allclose(empty.infection_frequency, 0.0)

    def test_correlation_adjacency_and_network_from_panel(self):
        adjacency = percolab.correlation_adjacency(CORR, 0.5)
        self.assertEqual(adjacency[0, 1], 0.9)
        self.assertEqual(adjacency[0, 2], 0.0)
        np.testing.assert_allclose(adjacency, adjacency.T)
        self.assertTrue(np.all(np.diag(adjacency) == 0.0))
        with self.assertRaises(ValueError):
            percolab.correlation_adjacency(
                np.array([[1.0, np.nan], [np.nan, 1.0]]), 0.5
            )

        rng = np.random.default_rng(2)
        factor = rng.normal(size=(200, 1))
        panel = factor + 0.3 * rng.normal(size=(200, 4))
        network = percolab.correlation_network(panel, 0.3)
        self.assertEqual(network.correlation.shape, (4, 4))
        self.assertEqual(network.components.labels.size, 4)
        self.assertEqual(network.mst.weight.size, 3)
        self.assertEqual(network.core_numbers.shape, (4,))
        self.assertTrue(np.isfinite(network.path_stats.efficiency))
        self.assertIn(network.path_stats.connected, (True, False))

    def test_path_statistics(self):
        path4 = PATH4
        stats = percolab.path_statistics(path4)
        self.assertAlmostEqual(stats.efficiency, 13.0 / 18.0, places=12)
        self.assertAlmostEqual(
            stats.average_path_length, 5.0 / 3.0, places=12
        )
        self.assertEqual(stats.diameter, 3)
        self.assertEqual(stats.n_reachable_pairs, 6)
        self.assertTrue(stats.connected)

        split = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        split_stats = percolab.path_statistics(split)
        self.assertAlmostEqual(split_stats.efficiency, 1.0 / 3.0, places=12)
        self.assertAlmostEqual(split_stats.average_path_length, 1.0)
        self.assertEqual(split_stats.diameter, 1)
        self.assertEqual(split_stats.n_reachable_pairs, 2)
        self.assertFalse(split_stats.connected)

        single = percolab.path_statistics(np.zeros((1, 1)))
        self.assertEqual(single.efficiency, 0.0)
        self.assertTrue(np.isnan(single.average_path_length))
        self.assertEqual(single.diameter, 0)
        self.assertEqual(single.n_reachable_pairs, 0)
        self.assertTrue(single.connected)

        with self.assertRaises(RFluxError):
            percolab.path_statistics(np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_core_numbers(self):
        path4 = PATH4
        np.testing.assert_array_equal(
            percolab.core_numbers(path4), [1, 1, 1, 1]
        )
        cycle4 = CYCLE4
        np.testing.assert_array_equal(
            percolab.core_numbers(cycle4), [2, 2, 2, 2]
        )
        complete = np.ones((4, 4)) - np.eye(4)
        np.testing.assert_array_equal(
            percolab.core_numbers(complete), [3, 3, 3, 3]
        )
        mixed = np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(
            percolab.core_numbers(mixed), [2, 2, 2, 0]
        )
        np.testing.assert_array_equal(
            percolab.core_numbers(np.zeros((2, 2))), [0, 0]
        )
        np.testing.assert_array_equal(
            percolab.core_numbers(np.zeros((1, 1))), [0]
        )

    def test_core_numbers_match_bruteforce(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            n = 6
            matrix = random_graph(rng, n)
            expected = np.zeros(n, dtype=np.int32)
            for k in range(1, n + 1):
                alive = set(range(n))
                while True:
                    removed = {
                        node
                        for node in alive
                        if sum(
                            1
                            for other in alive
                            if other != node and matrix[node, other] > 0.0
                        )
                        < k
                    }
                    if not removed:
                        break
                    alive -= removed
                for node in alive:
                    expected[node] = max(expected[node], k)
            np.testing.assert_array_equal(
                percolab.core_numbers(matrix), expected
            )

    def test_transitivity_assortativity_modularity(self):
        complete = np.ones((4, 4)) - np.eye(4)
        self.assertAlmostEqual(percolab.transitivity(complete), 1.0)
        path4 = PATH4
        self.assertAlmostEqual(percolab.transitivity(path4), 0.0)
        self.assertTrue(np.isnan(percolab.transitivity(np.zeros((4, 4)))))
        self.assertAlmostEqual(percolab.assortativity(path4), -0.5)
        star = np.zeros((4, 4))
        star[0, 1:] = 1.0
        star[1:, 0] = 1.0
        self.assertAlmostEqual(percolab.assortativity(star), -1.0)
        self.assertTrue(np.isnan(percolab.assortativity(complete)))
        self.assertTrue(np.isnan(percolab.transitivity(np.zeros((1, 1)))))
        self.assertTrue(np.isnan(percolab.assortativity(np.zeros((1, 1)))))

        two_triangles = np.zeros((6, 6))
        for i, j in ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)):
            two_triangles[i, j] = 1.0
            two_triangles[j, i] = 1.0
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        self.assertAlmostEqual(
            percolab.modularity(two_triangles, labels), 0.5
        )
        self.assertAlmostEqual(
            percolab.modularity(two_triangles, np.zeros(6, dtype=np.int32)),
            0.0,
        )
        self.assertAlmostEqual(
            percolab.modularity(complete, np.array([0, 0, 1, 1])),
            -1.0 / 6.0,
        )
        self.assertAlmostEqual(
            percolab.modularity(0.5 * complete, np.array([0, 0, 1, 1])),
            -1.0 / 6.0,
        )
        with self.assertRaises(ValueError):
            percolab.modularity(two_triangles, [0, 0, 0])
        with self.assertRaises(ValueError):
            percolab.modularity(two_triangles, [-1, 0, 0, 1, 1, 1])
        sparse_labels = np.array([0, 0, 0, 10, 10, 10])
        self.assertAlmostEqual(
            percolab.modularity(two_triangles, sparse_labels), 0.5
        )
        self.assertTrue(
            np.isnan(percolab.modularity(np.zeros((2, 2)), [0, 1]))
        )
        with self.assertRaises(RFluxError):
            percolab.modularity(
                np.array([[0.0, np.nan], [np.nan, 0.0]]), [0, 1]
            )

    def test_mst_clusters(self):
        distance = np.array(
            [
                [0.0, 1.0, 3.0, 4.0],
                [1.0, 0.0, 2.0, 5.0],
                [3.0, 2.0, 0.0, 1.0],
                [4.0, 5.0, 1.0, 0.0],
            ]
        )
        clusters = percolab.mst_clusters(distance, 1.5)
        self.assertEqual(clusters.n_components, 2)
        self.assertEqual(clusters.largest_size, 2)
        self.assertEqual(clusters.labels[0], clusters.labels[1])
        self.assertEqual(clusters.labels[2], clusters.labels[3])
        self.assertNotEqual(clusters.labels[0], clusters.labels[2])
        merged = percolab.mst_clusters(distance, 2.0)
        self.assertEqual(merged.n_components, 1)
        separated = percolab.mst_clusters(distance, 0.5)
        self.assertEqual(separated.n_components, 4)
        single = percolab.mst_clusters(np.zeros((1, 1)), 0.0)
        self.assertEqual(single.n_components, 1)
        self.assertEqual(single.largest_size, 1)

    def test_general_percolation_and_distribution(self):
        complete = np.ones((4, 4)) - np.eye(4)
        probabilities = [0.0, 1.0]
        baseline = percolab.bond_percolation(
            complete, probabilities, repetitions=20, seed=3
        )
        generic = percolab.percolation_scan(
            complete, probabilities, repetitions=20, seed=3
        )
        np.testing.assert_allclose(generic.giant_mean, baseline.giant_mean)
        np.testing.assert_allclose(generic.giant_std, baseline.giant_std)
        np.testing.assert_allclose(
            generic.susceptibility, baseline.susceptibility
        )

        node_weights = [1.0, 0.0, 0.0, 0.0]
        weighted = percolab.site_percolation_weighted(
            complete, node_weights, probabilities, repetitions=20, seed=3
        )
        generic_site = percolab.percolation_scan(
            complete,
            probabilities,
            kind="site",
            node_weights=node_weights,
            repetitions=20,
            seed=3,
        )
        np.testing.assert_allclose(
            generic_site.giant_mean, weighted.giant_mean
        )

        edge_weights = np.full((4, 4), 0.5)
        np.fill_diagonal(edge_weights, 0.0)
        separate = percolab.percolation_scan(
            complete,
            [0.0],
            edge_weights=edge_weights,
            repetitions=5,
            seed=1,
        )
        np.testing.assert_allclose(separate.giant_mean, [0.25])

        with self.assertRaises(ValueError):
            percolab.percolation_scan(
                complete, probabilities, kind="bond", node_weights=node_weights
            )
        with self.assertRaises(ValueError):
            percolab.percolation_scan(
                complete, probabilities, kind="site", edge_weights=edge_weights
            )
        with self.assertRaises(ValueError):
            percolab.percolation_scan(complete, probabilities, kind="other")
        with self.assertRaises(ValueError):
            percolab.percolation_scan(complete, [])

        bond_distribution = percolab.cluster_size_distribution(
            complete, 0.0, repetitions=5, seed=7
        )
        np.testing.assert_allclose(
            bond_distribution, [0.0, 4.0, 0.0, 0.0, 0.0]
        )
        full_distribution = percolab.cluster_size_distribution(
            complete, 1.0, repetitions=5, seed=7
        )
        np.testing.assert_allclose(
            full_distribution, [0.0, 0.0, 0.0, 0.0, 1.0]
        )
        site_distribution = percolab.cluster_size_distribution(
            complete, 0.0, kind="site", repetitions=5, seed=7
        )
        np.testing.assert_allclose(site_distribution, np.zeros(5))
        random_distribution = percolab.cluster_size_distribution(
            complete, 0.5, repetitions=50, seed=11
        )
        bond_mass = sum(
            size * random_distribution[size] for size in range(1, 5)
        )
        self.assertAlmostEqual(float(bond_mass), 4.0, places=12)
        site_random = percolab.cluster_size_distribution(
            complete, 0.5, kind="site", repetitions=50, seed=11
        )
        site_mass = sum(size * site_random[size] for size in range(1, 5))
        self.assertGreaterEqual(float(site_mass), 0.0)
        self.assertLessEqual(float(site_mass), 4.0)
        edge_binary = np.zeros((4, 4))
        edge_binary[0, 1] = edge_binary[1, 0] = 1.0
        weighted_distribution = percolab.cluster_size_distribution(
            complete, 1.0, edge_weights=edge_binary, repetitions=3, seed=7
        )
        np.testing.assert_allclose(
            weighted_distribution, [0.0, 2.0, 1.0, 0.0, 0.0]
        )
        site_weighted_distribution = percolab.cluster_size_distribution(
            complete,
            1.0,
            kind="site",
            node_weights=[1.0, 0.0, 0.0, 0.0],
            repetitions=3,
            seed=7,
        )
        np.testing.assert_allclose(
            site_weighted_distribution, [0.0, 1.0, 0.0, 0.0, 0.0]
        )
        with self.assertRaises(ValueError):
            percolab.cluster_size_distribution(complete, 1.5)

    def test_targeted_removal_and_core_percolation(self):
        complete = np.ones((4, 4)) - np.eye(4)
        curve = percolab.targeted_removal(
            complete, order=[0, 1, 2, 3], checkpoints=4
        )
        np.testing.assert_array_equal(curve.removed, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(
            curve.giant_fraction, [1.0, 0.75, 0.5, 0.25, 0.0]
        )
        np.testing.assert_allclose(curve.susceptibility, np.zeros(5))
        np.testing.assert_array_equal(curve.n_components, [1, 1, 1, 1, 0])

        path4 = PATH4
        path_curve = percolab.targeted_removal(
            path4, order=[1, 2, 0, 3], checkpoints=2
        )
        np.testing.assert_array_equal(path_curve.removed, [0, 2, 4])
        np.testing.assert_allclose(
            path_curve.giant_fraction, [1.0, 0.25, 0.0]
        )
        np.testing.assert_allclose(
            path_curve.susceptibility, [0.0, 0.25, 0.0]
        )

        for strategy in (
            "strength",
            "degree",
            "eigenvector",
            "betweenness",
            "core",
        ):
            order = percolab.removal_order(complete, strategy)
            np.testing.assert_array_equal(np.sort(order), np.arange(4))
        star = np.zeros((4, 4))
        star[0, 1:] = 1.0
        star[1:, 0] = 1.0
        self.assertEqual(int(percolab.removal_order(star, "degree")[0]), 0)
        self.assertEqual(int(percolab.removal_order(star, "strength")[0]), 0)

        with self.assertRaises(ValueError):
            percolab.targeted_removal(complete, order=[0, 0, 1, 2])
        with self.assertRaises(ValueError):
            percolab.targeted_removal(complete, checkpoints=0)
        dense = percolab.targeted_removal(
            np.zeros((2, 2)), checkpoints=4
        )
        np.testing.assert_array_equal(dense.removed, [0, 0, 1, 1, 2])

        core_curve = percolab.core_percolation(complete)
        np.testing.assert_array_equal(core_curve.k, [1, 2, 3])
        np.testing.assert_allclose(core_curve.core_fraction, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(core_curve.n_nodes, [4, 4, 4])
        empty_curve = percolab.core_percolation(np.zeros((3, 3)))
        self.assertEqual(empty_curve.k.size, 0)
        self.assertEqual(empty_curve.core_fraction.size, 0)

    def test_bridges_articulation_and_truss(self):
        path4 = PATH4
        np.testing.assert_array_equal(
            percolab.bridges(path4), [[0, 1], [1, 2], [2, 3]]
        )
        np.testing.assert_array_equal(
            percolab.articulation_points(path4), [False, True, True, False]
        )
        truss = percolab.truss_numbers(path4)
        self.assertEqual(truss.shape, (4, 4))
        self.assertEqual(truss.dtype, np.int32)
        for i, j in ((0, 1), (1, 2), (2, 3)):
            self.assertEqual(truss[i, j], 2)
            self.assertEqual(truss[j, i], 2)
        self.assertTrue(np.all(np.diag(truss) == 0))
        self.assertEqual(truss[0, 2], 0)

        cycle4 = CYCLE4
        self.assertEqual(percolab.bridges(cycle4).shape, (0, 2))
        self.assertFalse(np.any(percolab.articulation_points(cycle4)))

        triangle = np.ones((3, 3)) - np.eye(3)
        np.testing.assert_array_equal(
            percolab.truss_numbers(triangle), [[0, 3, 3], [3, 0, 3], [3, 3, 0]]
        )
        self.assertEqual(percolab.bridges(triangle).shape, (0, 2))

        complete4 = np.ones((4, 4)) - np.eye(4)
        np.testing.assert_array_equal(
            percolab.truss_numbers(complete4),
            [[0, 4, 4, 4], [4, 0, 4, 4], [4, 4, 0, 4], [4, 4, 4, 0]],
        )

        two_triangles = np.zeros((6, 6))
        for i, j in ((0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)):
            two_triangles[i, j] = two_triangles[j, i] = 1.0
        np.testing.assert_array_equal(
            percolab.bridges(two_triangles), [[2, 3]]
        )
        np.testing.assert_array_equal(
            percolab.articulation_points(two_triangles),
            [False, False, True, True, False, False],
        )
        joined_truss = percolab.truss_numbers(two_triangles)
        self.assertEqual(joined_truss[0, 1], 3)
        self.assertEqual(joined_truss[2, 3], 2)

        single = np.zeros((1, 1))
        self.assertEqual(percolab.bridges(single).shape, (0, 2))
        np.testing.assert_array_equal(
            percolab.articulation_points(single), [False]
        )
        np.testing.assert_array_equal(percolab.truss_numbers(single), [[0]])

        with self.assertRaises(RFluxError):
            percolab.bridges(np.array([[0.0, 1.0], [0.0, 0.0]]))
        with self.assertRaises(RFluxError):
            percolab.truss_numbers(np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_connectivity_and_truss_match_bruteforce(self):
        rng = np.random.default_rng(19)
        for _ in range(8):
            n = 6
            matrix = random_graph(rng, n)

            def components(skip_edge=None, skip_node=None):
                seen: set[int] = set()
                count = 0
                for start in range(n):
                    if start == skip_node or start in seen:
                        continue
                    count += 1
                    stack = [start]
                    seen.add(start)
                    while stack:
                        node = stack.pop()
                        for other in range(n):
                            if other == skip_node or other in seen:
                                continue
                            if matrix[node, other] <= 0.0:
                                continue
                            if skip_edge is not None and {
                                node,
                                other,
                            } == set(skip_edge):
                                continue
                            seen.add(other)
                            stack.append(other)
                return count

            base = components()
            expected_bridges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if matrix[i, j] > 0.0 and components((i, j)) > base:
                        expected_bridges.append((i, j))
            found_bridges = percolab.bridges(matrix)
            if expected_bridges:
                np.testing.assert_array_equal(
                    found_bridges, np.asarray(expected_bridges, dtype=np.uint32)
                )
            else:
                self.assertEqual(found_bridges.shape, (0, 2))
            expected_points = np.asarray(
                [components(skip_node=node) > base for node in range(n)],
                dtype=bool,
            )
            np.testing.assert_array_equal(
                percolab.articulation_points(matrix), expected_points
            )

            expected_truss = np.where(matrix > 0.0, 2, 0).astype(np.int32)
            for k in range(3, n + 1):
                alive = matrix > 0.0
                changed = True
                while changed:
                    changed = False
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not alive[i, j]:
                                continue
                            support = sum(
                                1
                                for w in range(n)
                                if w != i
                                and w != j
                                and alive[i, w]
                                and alive[j, w]
                            )
                            if support < k - 2:
                                alive[i, j] = alive[j, i] = False
                                changed = True
                expected_truss = np.where(
                    alive & (matrix > 0.0),
                    np.maximum(expected_truss, k),
                    expected_truss,
                ).astype(np.int32)
            np.testing.assert_array_equal(
                percolab.truss_numbers(matrix), expected_truss
            )

    def test_scan_summary_helpers(self):
        scan = percolab.bond_percolation(
            COMPLETE, [0.0, 1.0], repetitions=20, seed=3
        )
        self.assertEqual(scan.critical_probability(), 0.0)
        self.assertEqual(scan.p_half_giant(), 1.0)
        self.assertAlmostEqual(scan.max_susceptibility(), 0.75)
        self.assertAlmostEqual(scan.giant_area(), 0.625)
        empty = percolab.PercolationScan(
            probabilities=np.array([]),
            giant_mean=np.array([]),
            giant_std=np.array([]),
            susceptibility=np.array([]),
        )
        self.assertTrue(np.isnan(empty.critical_probability()))
        self.assertTrue(np.isnan(empty.p_half_giant()))
        self.assertTrue(np.isnan(empty.max_susceptibility()))
        self.assertEqual(empty.giant_area(), 0.0)

    def test_network_diagnostics(self):
        triangle = np.full((3, 3), 0.8)
        np.fill_diagonal(triangle, 1.0)
        network = percolab.network_from_correlation(triangle, 0.5)
        self.assertAlmostEqual(network.path_stats.efficiency, 1.0)
        self.assertAlmostEqual(network.path_stats.average_path_length, 1.0)
        self.assertEqual(network.path_stats.diameter, 1)
        self.assertTrue(network.path_stats.connected)
        np.testing.assert_array_equal(network.core_numbers, [2, 2, 2])
        self.assertEqual(network.max_core, 2)
        self.assertAlmostEqual(network.transitivity, 1.0)
        self.assertTrue(np.isnan(network.assortativity))

        network = percolab.network_from_correlation(CORR, 0.5)
        self.assertEqual(network.components.n_components, 2)
        self.assertFalse(network.path_stats.connected)
        self.assertAlmostEqual(network.path_stats.efficiency, 1.0 / 3.0)
        self.assertEqual(network.path_stats.n_reachable_pairs, 2)
        np.testing.assert_array_equal(network.core_numbers, [1, 1, 1, 1])
        self.assertEqual(network.max_core, 1)


    def test_monte_carlo_repeats_are_deterministic(self):
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(60, 60))
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 1.0)
        adjacency = np.maximum(matrix, 0.0)
        np.fill_diagonal(adjacency, 0.0)
        probabilities = np.linspace(0.1, 0.9, 7)

        first = percolab.site_percolation(
            adjacency, probabilities=probabilities, repetitions=11, seed=17
        )
        second = percolab.site_percolation(
            adjacency, probabilities=probabilities, repetitions=11, seed=17
        )
        np.testing.assert_array_equal(first.giant_mean, second.giant_mean)
        np.testing.assert_array_equal(first.giant_std, second.giant_std)
        np.testing.assert_array_equal(
            first.susceptibility, second.susceptibility
        )

        first_sizes = percolab.cluster_size_distribution(
            adjacency, 0.4, repetitions=9, seed=23
        )
        second_sizes = percolab.cluster_size_distribution(
            adjacency, 0.4, repetitions=9, seed=23
        )
        np.testing.assert_array_equal(first_sizes, second_sizes)


if __name__ == "__main__":
    unittest.main()
