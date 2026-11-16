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


if __name__ == "__main__":
    unittest.main()
