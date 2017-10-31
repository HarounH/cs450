#!/usr/bin/python
'''Solution for HW2.Q4, done without doing math.'''
__author__ = 'haroun habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import pdb
import numpy as np
import scipy.linalg as scla


def sgn(x):
    return (-1.0 if x < 0.0 else 1.0)


def householder(A):
    # A: Input, m x n matrix.
    # where m > n
    m = A.shape[0]
    n = A.shape[1]
    # assert(m == 5 and n == 4)
    i = 0
    eye = np.eye(m, dtype=np.float)
    Ai = A.copy()
    Q = np.eye(m)
    while i < n:
        ai = Ai[i:, i]
        ei = eye[i:, i]
        # compute v
        # nrm = np.linalg.norm(ai)
        # pdb.set_trace()
        v = ai + sgn(ai[0]) * ei * (np.linalg.norm(ai))
        matv = np.array([v.tolist()])
        Hi = np.eye(m - i) - (2 * np.transpose(matv) @ matv / np.dot(v, v))
        Hi = scla.block_diag(np.eye(i), Hi)
        Ai = np.matmul(Hi, Ai)

        Q = Hi @ Q
        print('v', i, '\n', v)
        print('H', i, '\n', Hi)
        print('A', i, '\n', Ai)

        i += 1


def get_givens(A, i, j):
    # returns G such that GA[i,j]=0.0
def count_givens(A):
    m = A.shape[0]
    n = A.shape[1]

A = np.array([[1, 2, -5, 1],
              [2, 0, 4, 5],
              [0, 3, -1, 2],
              [4, -1, 1, 5],
              [5, 2, 5, -2]],
             dtype=np.float)
# A = np.array([[1, -1, 4],
#               [1, 4, -2],
#               [1, 4, 2],
#               [1, -1, 0]],
#              dtype=np.float)
householder(A)
