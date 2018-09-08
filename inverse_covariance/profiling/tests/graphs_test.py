import numpy as np
import pytest

from inverse_covariance.profiling import (
    lattice,
    blocks,
    ClusterGraph,
    ErdosRenyiGraph,
    LatticeGraph,
)


class TestGraphs(object):
    def test_lattice_random_sign_false(self):
        n_features = 10
        alpha = 0.2
        prng = np.random.RandomState(1)
        adjacency = lattice(prng, n_features, alpha, random_sign=False)

        nnz = alpha * n_features

        # diagonal zero
        assert np.sum(adjacency[np.where(np.eye(n_features))]) == 0

        # all non-zeros are negative (random_sign=False)
        assert np.sum(adjacency[adjacency != 0] < 0) == np.sum(
            adjacency[adjacency != 0] != 0
        )

        # banded structure
        for nn in range(n_features):
            d = np.diag(np.ones((n_features - nn,)), k=nn)

            if nn == 0:
                continue

            elif nn <= nnz:
                assert np.sum(adjacency[np.where(d)]) != 0

            else:
                assert np.sum(adjacency[np.where(d)]) == 0

    def test_lattice_random_sign_true(self):
        n_features = 10
        alpha = 0.2
        prng = np.random.RandomState(1)
        adjacency = lattice(prng, n_features, alpha, random_sign=True)

        nnz = alpha * n_features

        # diagonal zero
        assert np.sum(adjacency[np.where(np.eye(n_features))]) == 0

        # some non-zeros are negative (random_sign=True)
        assert np.sum(adjacency[adjacency != 0] < 0) != np.sum(
            adjacency[adjacency != 0] != 0
        )

        # banded structure
        for nn in range(n_features):
            d = np.diag(np.ones((n_features - nn,)), k=nn)

            if nn == 0:
                continue

            elif nn <= nnz:
                assert np.sum(adjacency[np.where(d)]) != 0

            else:
                assert np.sum(adjacency[np.where(d)]) == 0

    def test_blocks_chain_blocks_false(self):
        prng = np.random.RandomState(1)
        n_block_features = 4
        n_blocks = 3
        block = np.ones((n_block_features, n_block_features))

        adjacency = blocks(prng, block, n_blocks=n_blocks, chain_blocks=False)

        n_features = n_block_features * n_blocks

        assert adjacency.shape[0] == n_features

        # diagonal zero
        assert np.sum(adjacency[np.where(np.eye(n_features))]) == 0

        # for each block, assert sum of the blocks minus the diagonal
        for nn in range(n_blocks):
            brange = slice(nn * n_block_features, (nn + 1) * n_block_features)
            assert (
                np.sum(adjacency[brange, brange])
                == n_block_features ** 2 - n_block_features
            )

        # assert that all nonzeros equal sum of all blocks above
        assert (
            np.sum(adjacency.flat)
            == (n_block_features ** 2 - n_block_features) * n_blocks
        )

    def test_blocks_chain_blocks_true(self):
        prng = np.random.RandomState(1)
        n_block_features = 4
        n_blocks = 3
        block = np.ones((n_block_features, n_block_features))

        adjacency = blocks(prng, block, n_blocks=n_blocks, chain_blocks=True)

        n_features = n_block_features * n_blocks

        assert adjacency.shape[0] == n_features

        # diagonal zero
        assert np.sum(adjacency[np.where(np.eye(n_features))]) == 0

        # for each block, assert sum of the blocks minus the diagonal
        for nn in range(n_blocks):
            brange = slice(nn * n_block_features, (nn + 1) * n_block_features)
            assert (
                np.sum(adjacency[brange, brange])
                == n_block_features ** 2 - n_block_features
            )

        # assert that all nonzeros DO NOT equal sum of all blocks above
        assert (
            np.sum(adjacency.flat)
            != (n_block_features ** 2 - n_block_features) * n_blocks
        )

    @pytest.mark.parametrize(
        "graph,seed",
        [
            (ClusterGraph(), None),
            (ErdosRenyiGraph(), None),
            (LatticeGraph(), None),
            (LatticeGraph(random_sign=True, chain_blocks=False, seed=3), 3),
            (ClusterGraph(n_blocks=5, seed=3), 3),
            (ErdosRenyiGraph(seed=3), 3),
        ],
    )
    def test_graph_classes(self, graph, seed):
        """Simple smell test on inidivudal graph classes"""
        n_features = 50
        alpha = 0.1
        covariance, precision, adjacency = graph.create(n_features, alpha)
        assert covariance.shape[0] == covariance.shape[1]
        assert covariance.shape[0] == n_features
        assert precision.shape[0] == precision.shape[1]
        assert precision.shape[0] == n_features
        assert adjacency.shape[0] == adjacency.shape[1]
        assert adjacency.shape[0] == n_features
        assert np.sum(covariance.flat) > 0
        assert np.sum(precision.flat) > 0
        assert np.sum(adjacency.flat) > 0
        if seed is not None:
            assert graph.seed == seed
