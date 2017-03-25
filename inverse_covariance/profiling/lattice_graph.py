from __future__ import absolute_import

from .graphs import Graph, lattice


class LatticeGraph(Graph):
    """Returns the adjacency matrix for a lattice/banded network via .create().

    The graph can be made fully connected using chaining assumption when
    chain_blocks=True (default).

    Parameters
    -----------
    random sign : bool (default=False)
        Randomly modulate each entry by 1 or -1 with probability of 1/2.

    low : float (0, 1) (default=0.3)
        Lower bound for np.random.RandomState.uniform before normalization.

    high : float (0, 1) > low (default=0.7)
        Upper bound for np.random.RandomState.uniform before normalization.

    n_blocks : int (default=2)
        Number of blocks.  Returned matrix will be square with
        shape n_block_features * n_blocks.

    chain_blocks : bool (default=True)
        Apply random lattice structure to chain blocks together.

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, random_sign=False, low=0.7, high=0.7, **kwargs):
        self.random_sign = random_sign
        self.low = low
        self.high = high
        super(LatticeGraph, self).__init__(**kwargs)

    def prototype_adjacency(self, n_block_features, alpha):
        """Build a new graph.

        Doc for ".create(n_features, alpha)"

        Parameters
        -----------
        n_features : int

        alpha : float (0,1)
            The complexity / sparsity factor.

            Each graph will have a minimum of

                n_blocks * ceil(alpha * n_block_features)

                where

                n_block_features = floor(n_features / self.n_blocks)

            edges and exactly this amount if chain_blocks=False.

        Returns
        -----------
        (n_features, n_features) matrices: covariance, precision, adjacency
        """
        return lattice(self.prng, n_block_features, alpha,
                       random_sign=self.random_sign,
                       low=self.low,
                       high=self.high)
