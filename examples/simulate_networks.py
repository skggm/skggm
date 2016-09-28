import numpy as np
import scipy as sp
from scipy import linalg

def _new_graph(n_features, alpha, adj_type='erdos-renyi', random_sign=False):
    global prng
    prng = np.random.RandomState(1)
    
    if adj_type=='erdos-renyi':
        adjacency = _random_er_network(n_features=n_features,alpha=alpha,random_state=prng)
    elif adj_type=='banded':
        adjacency = _lattice_network(n_features=n_features,alpha=alpha,random_sign=random_sign,random_state=prng)
    elif adj_type=='cluster':
        if n_features<=25:
            n_groups=3
        else:
            n_groups=5            
        adjacency = _cluster_network(n_features=n_features,n_groups=n_groups,alpha=alpha,adj_type='cluster',random_sign=random_sign,random_state=prng)
    else:
        adjacency = _random_er_network(n_features=n_features,alpha=alpha,random_state=prng)

    prec = _make_diag_dominant(adjacency)
    prec = _make_correlation(prec)
    cov = np.linalg.inv(prec)
    cov = _make_correlation(cov)
    return cov, prec, adjacency

def _random_er_network(n_features, alpha,random_state=np.random.RandomState(1)):
    adj = make_sparse_spd_matrix(n_features,
                                  alpha=alpha, # prob that a coeff is zero
                                  smallest_coef=0.7,
                                  largest_coef=0.7,
                                  random_state=random_state)
    return adj

def _lattice_network(n_features, alpha =.3, random_sign=False,random_state=np.random.RandomState(1)):
    """
    Creates a lattice network of size n_features x n_features. Each row has maximum edges of ceil(.3*n_features)
    """
    
    adj = np.zeros([n_features,n_features])
    row = np.zeros([n_features])
    degree = 1+np.round(alpha*n_features/2)
    if random_sign:
        sign_row = -1.0*np.ones(degree)+ 2*(random_state.uniform(low=0,high=1,size=int(degree))>.5)
    else:
        sign_row = -1.0*np.ones(degree)
    row[1:1+degree] = sign_row*random_state.uniform(low=.3, high=.7, size=int(degree))
    row /= abs(sum(row))
    adj = sp.linalg.toeplitz(c=row,r=row)
    return adj

def _hub_network(n_features, alpha,random_state=np.random.RandomState(1)):
    
    adj = np.zeros([n_features,n_features])
        
    return adj

def _smallw_network(n_features, alpha,random_state=np.random.RandomState(1)):
    """
    Create disjoint groups of variables (e.g n_features=15, 5 groups of size 3). 
    Make fully connected using chain assumption.  
    """
    adj = np.zeros([n_features,n_features])
        
    return adj

def _cluster_network(n_features, n_groups, alpha, random_sign=False,adj_type='banded',chain_blocks=True,random_state=np.random.RandomState(1)):
    """
    Create disjoint groups of variables (e.g n_features=15, 5 groups of size 3). 
    Make fully connected using chaining assumption.  
    """
    adj = np.zeros([n_features,n_features])    
    n_block = int(np.floor(n_features/n_groups))
    
    if adj_type=='banded':
        small_adj = _lattice_network(n_features=n_block,alpha=alpha,random_sign=random_sign,random_state=random_state)
    elif adj_type=='cluster':
        small_adj = -np.ones((n_block,n_block))*.5 + random_state.uniform(low=.05, high=.2, size=(n_block,n_block))
    else:
        small_adj = _lattice_network(n_features=n_block,alpha=alpha,random_sign=random_sign,random_state=random_state) 
    
    if chain_blocks:
        dep_groups = _lattice_network(n_features=n_groups,alpha=round(.01+0.5/n_groups,2),random_sign=False,random_state=random_state)
        dep_groups *= -.1
        dep_groups += np.eye(n_groups)
    else:
        dep_groups = np.eye(n_groups)
        
    adj = np.kron(dep_groups, small_adj)
    adj[np.where(np.eye(n_features))] = 0
    
    return adj


def _make_diag_dominant(adjacency):    
    
    d = np.diag(np.sum(np.abs(adjacency),axis=1)+.01)
    adjacency += d
    return adjacency

def _make_correlation(adjacency):
    """
    Call only after diagonal dominance is ensured. 
    TODO: Check for diagonally dominant adjacency first. 
    """   
    d = np.sqrt(np.diag(adjacency))
    adjacency /= d
    adjacency /= d[:, np.newaxis]
    return adjacency
