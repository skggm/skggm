import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, int32_t

cdef extern from "QUIC.h":
    void QUIC(char mode, uint32_t& p, double* S, double* Lambda0,
	  uint32_t& pathLen, double* path, double& tol,
	  int32_t& msg, uint32_t& maxIter,
	  double* X, double* W, double* opt, double* cputime,
	  uint32_t* iter, double* dGap)

def quic(char* mode, int p, 
        np.ndarray[np.float64_t, ndim=2, mode='c'] S, 
        np.ndarray[np.float64_t, ndim=2, mode='c'] L,
        int pathLen,
        np.ndarray[np.float64_t, ndim=1, mode='c'] path,
        double tol, int msg, int max_iter, 
        np.ndarray[np.float64_t, ndim=2, mode='c'] X, 
        np.ndarray[np.float64_t, ndim=2, mode='c'] W,
        np.ndarray[np.float64_t, ndim=1, mode='c'] opt,
        np.ndarray[np.float64_t, ndim=1, mode='c'] cputime,
        np.ndarray[np.uint32_t, ndim=1, mode='c'] iter,
        np.ndarray[np.float64_t, ndim=1, mode='c'] dGap
        ):
    """
    """
    cdef uint32_t _p = p
    cdef uint32_t _pathLen = pathLen
    cdef int32_t _msg = msg
    cdef uint32_t _max_iter = max_iter
    cdef uint32_t* _iter = <uint32_t*> iter.data
    QUIC(mode[0], _p, &S[0,0], &L[0,0], _pathLen, &path[0], tol, _msg, _max_iter, 
            &X[0,0], &W[0,0], &opt[0], &cputime[0], &_iter[0], &dGap[0])

    return
