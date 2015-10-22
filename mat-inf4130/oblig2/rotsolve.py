from __future__ import division
from numpy import *
import numpy.linalg as linalg
from scipy.linalg import hilbert
import matplotlib.pyplot as plt

def rbacksolve(A, b, d):
    """
    Solve the system A*x = b for an upper triangular d-banded matrix A,
    using algorithm 2.7 (row-oriented backward substitution).
    The input vector b will be updated with the solution x (i.e. in-place).
    
    A: An upper triangular matrix
    b: The right hand side
    d: The band-width
    """
    n = len(b)
    b[n - 1] /= A[n - 1,n - 1]
    for k in range(n-2,-1,-1):
        uk = array([n, k + d + 1]).min()
        b[k] = b[k] - dot(A[k,(k+1):uk], b[(k+1):uk])
        b[k] /= A[k,k]
    

def rotsolveAlt(A, b):
    """
    Solve the system Ax=b where A any N x N matrix, again using Givens
    rotations. Modified from rothesstri(A, b).
    
    A: A matrix. Nonsingular ones preferred, obviously.
    b: The right hand side.
    """
    n = shape(A)[0]
    A = hstack([A, b])
    for k in xrange(n-1):   # columns
        for j in xrange(k + 1, n):   # rows
            r = linalg.norm([ A[k , k] , A[j, k] ], 2)
            if r>0:
                c=A[k, k]/r; s=A[j, k]/r
                A[[k, j],(k + 1):(n+1)] = \
                    mat([[c, s],[-s, c]])* \
                    A[[k, j],(k + 1):(n+1)]
            A[k, k] = r; A[j,k] = 0
    z = A[:, n].copy()
    rbacksolve(A[:, :n], z, n)
    return z
    
def rotsolve(A, b):
    """
    Solve the system Ax=b where A any N x N matrix, again using Givens
    rotations. Modified from rothesstri(A, b).
    
    A: A matrix. Nonsingular ones preferred, obviously.
    b: The right hand side.
    """
    n = shape(A)[0]
    A = hstack([A, b])
    for k in xrange(n-1):   # columns
        for j in xrange(n - 2, k -1, -1):   # rows
            r = linalg.norm([ A[j , k] , A[j + 1, k] ], 2)
            if r>0:
                c=A[j, k]/r; s=A[j + 1, k]/r
                A[[j, j + 1],(k + 1):(n+1)] = \
                    mat([[c, s],[-s, c]])* \
                    A[[j, j + 1],(k + 1):(n+1)]
            A[j, k] = r; A[j+1,k] = 0
    z = A[:, n].copy()
    rbacksolve(A[:, :n], z, n)
    return z
    
if __name__ == '__main__':
    
    N = 20
    
    H_ = hilbert(N)
    xe_ = mat(ones(N)).T
    err = empty(N)
    errAlt = empty(N)
    iterations = arange(N)
    for n in iterations:
#        H = H_[:n+1, :n+1]; xe = xe_[:n+1]
        H = hilbert(n+1); xe = mat(ones(n+1)).T
        b = H*xe
        x = rotsolve(H, b)
        err[n] = linalg.norm(x - xe, 2)
        y = rotsolveAlt(H, b)
        errAlt[n] = linalg.norm(y - xe, 2)
    
    plt.plot(1+iterations, err, 'b')
    plt.plot(1 +iterations, errAlt, 'r')
    plt.legend(['Algorithm 1','Algorithm 2'])
    plt.xlabel('size of matrix [n]')
    plt.ylabel('||x - x_e||_2')
    plt.show()
