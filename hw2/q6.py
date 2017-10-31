#!/usr/bin/python
'''HW2.Q6 solution for CS450 Fall 2017'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import pdb

# A,z defined elsewhere
# A = np.random.rand((4,2))
# z = np.random.rand((4,1))

m = A.shape[0]
n = A.shape[1]
h = 1.0 / (m + 1)
W1 = np.eye(m)
W2 = np.diag([1 + i for i in range(0, m)])
W3 = np.diag([2 * (m + 1) * (m + 1)] * m) \
    + np.diag([-(m + 1) * (m + 1)] * (m - 1), -1) \
    + np.diag([-(m + 1) * (m + 1)] * (m - 1), 1)


def solve(W, A, z):
    At = np.transpose(A)
    return (A @ (np.linalg.inv(At @ W @ A)) @ At @ z)


y1 = solve(W1, A, z)
y2 = solve(W2, A, z)
y3 = solve(W3, A, z)
pdb.set_trace()
