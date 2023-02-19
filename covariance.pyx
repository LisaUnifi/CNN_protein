import numpy as np
cimport numpy as np


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def with_cython(np.ndarray[DTYPE_t, ndim=2] msa):
    assert msa.dtype == DTYPE 

    cdef int size0 = msa.shape[0]
    cdef int size1 = msa.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] weight = np.ones(size0, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] pa = np.ones([size1, 21], dtype=DTYPE) 
    cdef np.ndarray[DTYPE_t, ndim=4] pab = np.zeros([size1, size1, 21, 21], dtype=DTYPE) 
    cdef np.ndarray[DTYPE_t, ndim=4] cov = np.zeros([size1, size1, 21, 21], dtype=DTYPE) 
    cdef np.ndarray[DTYPE_t, ndim=3] cov_final = np.zeros([size0, size1, 441], dtype=DTYPE) 
    cdef int i, j, similar, a, k, b
    cdef float meff

    for i in range(size0):
        for j in range(i+1, size0):
            similar = int(size1 * 0.2)
            similar = similar - (sum(msa[i, :] != msa[j, :]))
            if similar > 0:
                weight[i] += 1
                weight[j] += 1

    #calcolo Meff 
    weight = 1/weight
    meff = np.sum(weight)

    #calcolo frequenza
    for i in range(size1):
        for a in range(21):
            pa[i, a] = 1.0 
        for k in range(size0):
            a = int(msa[k, i])
            if a < 21:
                pa[i, a] = pa[i, a] + weight[k]
        for a in range(21):
            pa[i, a] = pa[i, a] / (21.0 + meff)
            
    for i in range(size1):
        for j in range(size1):
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = 1.0 / 21.0
            for k in range(size0):
                a = int(msa[k, i])
                b = int(msa[k, j])
                if (a < 21 and b < 21):
                    pab[i, j, a, b] = pab[i, j, a, b] + weight[k]
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = pab[i, j, a, b] / (21.0 + meff)

    #calcolo matrice di covarianza
    for a in range(21):
        for b in range(21):
            for i in range(size1):
                for j in range(size1):
                    cov[i, j, a, b] = pab[i, j, a, b] - (pa[i, a] * pa[j, b])
    cov_final = cov.reshape(size1, size1, 21*21)

    return cov_final