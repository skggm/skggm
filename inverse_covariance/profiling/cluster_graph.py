from __future__ import absolute_import

import numpy as np
from .graphs import Graph


class ClusterGraph(Graph):
    """Returns the adjacency matrix for a cluster network via .create().

    The graph can be made fully connected using chaining assumption when
    chain_blocks=True (default).

    Parameters
    -----------
    low : float (0, 1) (default=0.3)
        Lower bound for np.random.RandomState.uniform cluster values.

    high : float (0, 1) > low (default=0.7)
        Upper bound for np.random.RandomState.uniform vluster values.

    n_blocks : int (default=2)
        Number of blocks.  Returned matrix will be square with
        shape n_block_features * n_blocks.

    chain_blocks : bool (default=True)
        Apply random lattice structure to chain blocks together.

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, low=0.7, high=0.7, **kwargs):
        self.low = low
        self.high = high
        super(ClusterGraph, self).__init__(**kwargs)

    def prototype_adjacency(self, n_block_features, alpha=None):
        """Build a new graph.

        Doc for ".create(n_features, alpha)"

        Parameters
        -----------
        n_features : int

        [alpha] : float (0,1)
            Unused.

        Each graph will have a minimum of

            (n_blocks * n_block_features**2 - n_blocks) / 2

        edges and exactly this amount if chain_blocks=False.

        Returns
        -----------
        (n_features, n_features) matrices: covariance, precision, adjacency
        """
        return (-np.ones((n_block_features, n_block_features)) * 0.5 +
                self.prng.uniform(low=self.low, high=self.high,
                                  size=(n_block_features, n_block_features)))
