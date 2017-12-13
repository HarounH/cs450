'''
    Utility functions for other computer problems.
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import pdb

def define_poisson_system(As, bs, x0s, n=20, analyze=False):
    A = sp.diags([1, 1, 1, 1, 1, 1, -6.0], [1, -1, -n, n, -n**2, n**2, 0], shape=(n**3, n**3))
    As.append(A)
    bs.append((n**-2) * np.random.rand(n**3))
    x0s.append(np.random.rand(n**3))
    if analyze:
        pdb.set_trace()


def define_heat(As, bs, x0s, a=0.1, n=20, analyze=False):
    sz = n**3
    n2 = n**2
    A = sp.csr_matrix((sz, sz), dtype=float)
    b = np.zeros(sz)
    for w in range(sz):
        i = w / n2
        j = (w % n2) / n
        k = w % n
        if k == 0:
            A[w, w] = 1
            b[w] = np.random.rand()
        elif i == 0 or j == 0:
            A[w, w] = 1
        elif i == (n - 1) or j == (n - 1):
            A[w, w] = 1
        else:
            if (w + 1) < sz:
                A[w, w + 1] = 1 / (2 * a * n)
            if (w - 1) > 0:
                A[w, w - 1] = -1 / (2 * a * n)
            if (w + n) < sz:
                A[w, w + n] = 1
            if (w - n) > 0:
                A[w, w - n] = 1
            if (w + n2) < sz:
                A[w, w + n2] = 1
            if (w - n2) > 0:
                A[w, w - n2] = 1
            A[w, w] = -4
    As.append(A)
    bs.append(b)
    x0s.append(np.random.rand(sz))
    if analyze:
        pdb.set_trace()

def generate_matrix(sz):
    '''
        ARGS
        -------
        sz: int, desired size of matrix
        RETURNS
        -------
        A : positive definite symmetric matrix of size sz
    '''
    A = np.random.rand(sz, sz)
    A = 0.5 * (A + A.T)
    A = A + sz * np.eye(sz)
    return A


def condition_matrix(A, c):
    '''
        ARGS
        ------
        A: n*n matrix, symmetric positive definite
        c: float, desired condition number (ratio of singular values)

        RETURNS
        ------
        B: A +kI where k is chosen such that condition number is c
    '''
    n = A.shape[0]
    eigs = sorted(np.linalg.eig(A)[0])
    lambda1 = eigs[0]
    lambdan = eigs[-1]
    sqrtc = np.sqrt(c)
    return A + np.eye(n) * (lambda1 - sqrtc * lambdan) / (sqrtc - 1)


if __name__ == '__main__':
    define_heat([], [], [], n=4, analyze=True)
