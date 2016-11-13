import numpy as np 
import scipy as sp
from sklearn.datasets import make_sparse_spd_matrix



def lattice(prng, n_features, alpha, random_sign=False):
    """Returns the adjacency matrix for a lattice network. 
    Each row has maximum edges of np.ceil(alpha * n_features).

    Parameters
    -----------        
    n_features : int 

    alpha : float (0,1) 
        The complexity / sparsity factor.

    random sign : bool (default=False)
    """
    U_LOW = 0.3
    U_HIGH = 0.7

    adj = np.zeros((n_features, n_features))
    row = np.zeros((n_features,))
    degree = int(1 + np.round(alpha * n_features / 2.))

    if random_sign:
        sign_row = (-1.0 * np.ones(degree) + 
                    2 * (prng.uniform(low=0, high=1, size=degree) > .5))
    else:
        sign_row = -1.0 * np.ones(degree)
    
    # QUESTION: are we indexing starting by 1 intentionally here or is this a 
    #           holdover error from matlab?
    row[1: 1 + degree] = sign_row * prng.uniform(low=U_LOW, high=U_HIGH,
                                                 size=degree)
    row /= np.abs(np.sum(row))
    return sp.linalg.toeplitz(c=row, r=row)


def cluster(prng, n_features, alpha, n_groups=2, random_sign=False, 
            adj_type='cluster', chain_blocks=True):
    """Returns the adjacency matrix for a cluster network.

    This function creates disjoint groupf of variables, where each group is 
    size n_features/n_groups.  The graph can be made fully connected using 
    chaining assumption when chain_blocks=True (default).
    
    Parameters
    -----------        
    n_features : int 

    alpha : float (0,1) 
        The complexity / sparsity factor.

    n_groups : 

    random_sign : bool (default=False)

    adj_type : 'banded' or 'cluster' (default='cluster')

    chain_blocks : bool (default=True)
    
    """
    U_LOW = 0.05
    U_HIGH = 0.2

    adj = np.zeros((n_features, n_features))    
    n_block = int(np.floor(1. * n_features / n_groups))
    
    if adj_type == 'banded':
        block_adj = lattice(prng, n_block, alpha, random_sign=random_sign)
    elif adj_type == 'cluster':
        block_adj = (-np.ones((n_block, n_block)) * 0.5 + 
                     prng.uniform(low=U_LOW, high=U_HIGH, size=(n_block, n_block)))
    else:
        raise ValueError("adj_type must be 'banded' or 'cluster'")
        return
    
    dep_groups = np.eye(n_groups)
    if chain_blocks:
        chain_alpha = np.round(0.01 + 0.5 / n_groups, 2)
        chain_groups = lattice(prng, n_groups, chain_alpha, random_sign=False)
        chain_groups *= -0.1
        dep_groups += chain_groups
        
    adj = np.kron(dep_groups, block_adj)
    adj[np.where(np.eye(n_features))] = 0
    return adj


# def hub()

# def small_world()


class Graph(object):
    """Returns the adjacency matrix for a specified network via the .sample()
    method.

    Parameters
    ----------- 
    network_type : one of 
        'erdos-renyi' (default)
        'lattice'
        'cluster'
       

    network_kwargs : dict of args for adjacency functions
        network_type: 'lattice'
        set keys: 'random_sign' 

        network_type: 'cluster'
        set keys: 'adj_type', 'random_sign', 'n_groups', 'chain_blocks'

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, network_type='erdos-renyi', network_kwargs={}, seed=1):
        self.network_type = network_type
        self.network_kwargs = network_kwargs
        self.prng = np.random.RandomState(seed)

    def _make_diagonally_dominant(self, adjacency):    
        adjacency += np.diag(np.sum(np.abs(adjacency), axis=1) + 0.01)
        return adjacency

    def _make_correlation(self, adjacency):
        """Call only after diagonal dominance is ensured. 
        TODO: Check for diagonally dominant adjacency first. 
        """   
        d = np.sqrt(np.diag(adjacency))
        adjacency /= d
        adjacency /= d[:, np.newaxis]
        return adjacency

    def sample(self, n_features, alpha):
        """Draw a new graph.

        Parameters
        -----------        
        n_features : int 

        alpha : float (0,1) 
            The complexity / sparsity factor for each graph type.
        
        Returns
        -----------  
        covariance : 

        precision : 

        adjacency : 
        """
        if self.network_type == 'erdos-renyi':
            SPD_LOW = 0.7
            SPD_HIGH = 0.7
            adjacency = make_sparse_spd_matrix(n_features,
                                               alpha=np.abs(1.0 - alpha), 
                                               smallest_coef=SPD_LOW,
                                               largest_coef=SPD_HIGH,
                                               random_state=self.prng)

        elif self.network_type == 'lattice':
            adjacency = lattice(self.prng, n_features, alpha, **self.kwargs)        

        elif self.network_type == 'cluster':
            if n_features <= 25 and self.n_groupds > 3:
                print 'Warning, number of groups is too large for n_features'           
            
            adjacency = cluster(self.prng, n_features, alpha, **self.kwargs)
        else:
            raise ValueError(("network_type must be 'erdos-renyi', 'lattice', ",
                              "or 'cluster'"))
            return

        precision = self._make_diagonally_dominant(adjacency)
        precision = self._make_correlation(precision)
        covariance = np.linalg.inv(precision)
        covariance = self._make_correlation(covariance)
        return covariance, precision, adjacency

