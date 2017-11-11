import numpy as np


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
