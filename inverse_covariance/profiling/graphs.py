import numpy as np
import scipy as sp


def lattice(prng, n_features, alpha, random_sign=False, low=0.3, high=0.7):
    """Returns the adjacency matrix for a lattice network.

    The resulting network is a Toeplitz matrix with random values summing
    between -1 and 1 and zeros along the diagonal.

    The range of the values can be controlled via the parameters low and high.
    If random_sign is false, all entries will be negative, otherwise their sign
    will be modulated at random with probability 1/2.

    Each row has maximum edges of np.ceil(alpha * n_features).

    Parameters
    -----------
    n_features : int

    alpha : float (0, 1)
        The complexity / sparsity factor.

    random sign : bool (default=False)
        Randomly modulate each entry by 1 or -1 with probability of 1/2.

    low : float (0, 1) (default=0.3)
        Lower bound for np.random.RandomState.uniform before normalization.

    high : float (0, 1) > low (default=0.7)
        Upper bound for np.random.RandomState.uniform before normalization.
    """
    degree = int(1 + np.round(alpha * n_features / 2.))

    if random_sign:
        sign_row = (-1.0 * np.ones(degree) +
                    2 * (prng.uniform(low=0, high=1, size=degree) > .5))
    else:
        sign_row = -1.0 * np.ones(degree)

    # in the *very unlikely* event that we draw a bad row that sums to zero
    # (which is only possible when random_sign=True), we try again up to
    # MAX_ATTEMPTS=5 times.  If we are still unable to draw a good set of
    # values something is probably wrong and we raise.
    MAX_ATTEMPTS = 5
    attempt = 0
    row = np.zeros((n_features,))
    while np.sum(row) == 0 and attempt < MAX_ATTEMPTS:
        row = np.zeros((n_features,))
        row[1: 1 + degree] = sign_row * prng.uniform(low=low,
                                                     high=high,
                                                     size=degree)
        attempt += 1

    if np.sum(row) == 0:
        raise Exception('InvalidLattice', 'Rows sum to 0.')
        return

    # sum-normalize and keep signs
    row /= np.abs(np.sum(row))

    return sp.linalg.toeplitz(c=row, r=row)


def blocks(prng, block, n_blocks=2, chain_blocks=True):
    """Replicates `block` matrix n_blocks times diagonally to create a
    square matrix of size n_features = block.size[0] * n_blocks and with zeros
    along the diagonal.

    The graph can be made fully connected using chaining assumption when
    chain_blocks=True (default).

    This utility can be used to generate cluster networks or banded lattice n
    networks, among others.

    Parameters
    -----------
    block : 2D array (n_block_features, n_block_features)
        Prototype block adjacency matrix.

    n_blocks : int (default=2)
        Number of blocks.  Returned matrix will be square with
        shape n_block_features * n_blocks.

    chain_blocks : bool (default=True)
        Apply random lattice structure to chain blocks together.
    """
    n_block_features, _ = block.shape
    n_features = n_block_features * n_blocks
    adjacency = np.zeros((n_features, n_features))

    dep_groups = np.eye(n_blocks)
    if chain_blocks:
        chain_alpha = np.round(0.01 + 0.5 / n_blocks, 2)
        chain_groups = lattice(prng, n_blocks, chain_alpha, random_sign=False)
        chain_groups *= -0.1
        dep_groups += chain_groups

    adjacency = np.kron(dep_groups, block)
    adjacency[np.where(np.eye(n_features))] = 0
    return adjacency


def _to_diagonally_dominant(mat):
    """Make matrix unweighted diagonally dominant using the Laplacian."""
    mat += np.diag(np.sum(mat != 0, axis=1) + 0.01)
    return mat


def _to_diagonally_dominant_weighted(mat):
    """Make matrix weighted diagonally dominant using the Laplacian."""
    mat += np.diag(np.sum(np.abs(mat), axis=1) + 0.01)
    return mat


def _rescale_to_unit_diagonals(mat):
    """Rescale matrix to have unit diagonals.

    Note: Call only after diagonal dominance is ensured.
    """
    d = np.sqrt(np.diag(mat))
    mat /= d
    mat /= d[:, np.newaxis]
    return mat


class Graph(object):
    """Base class that returns the adjacency matrix for a network via
    the .create() method.

    Parameters
    -----------
    n_blocks : int (default=2)
        Number of blocks.  Returned matrix will be square with
        shape n_block_features * n_blocks.

    chain_blocks : bool (default=True)
        Apply random lattice structure to chain blocks together.

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, n_blocks=2, chain_blocks=True, seed=1):
        self.n_blocks = n_blocks
        self.chain_blocks = chain_blocks
        self.seed = seed
        self.prng = np.random.RandomState(self.seed)

        if n_blocks == 1 and chain_blocks:
            raise ValueError("Requires chain_blocks=False when n_blocks=1.")
            return

    def to_precision(self, adjacency, weighted=True, rescale=True):
        if weighted:
            dd_adj = _to_diagonally_dominant_weighted(adjacency)
        else:
            dd_adj = _to_diagonally_dominant(adjacency)

        if rescale:
            return _rescale_to_unit_diagonals(dd_adj)

        return dd_adj

    def to_covariance(self, precision, rescale=True):
        covariance = np.linalg.inv(precision)

        if rescale:
            return _rescale_to_unit_diagonals(covariance)

        return covariance

    def prototype_adjacency(self, n_block_features, alpha):
        """Override this method with a custom base graph."""
        pass

    def create(self, n_features, alpha):
        """Build a new graph with block structure.

        Parameters
        -----------
        n_features : int

        alpha : float (0,1)
            The complexity / sparsity factor for each graph type.

        Returns
        -----------
        (n_features, n_features) matrices: covariance, precision, adjacency
        """
        n_block_features = int(np.floor(1. * n_features / self.n_blocks))
        if n_block_features * self.n_blocks != n_features:
            raise ValueError(
                ('Error: n_features {} not divisible by n_blocks {}.'
                 'Use n_features = n_blocks * int').format(
                    n_features,
                    self.n_blocks)
                )
            return

        block_adj = self.prototype_adjacency(n_block_features, alpha)
        adjacency = blocks(self.prng,
                           block_adj,
                           n_blocks=self.n_blocks,
                           chain_blocks=self.chain_blocks)

        precision = self.to_precision(adjacency)
        covariance = self.to_covariance(precision)
        return covariance, precision, adjacency
