#!/usr/bin/python
'''
    This file implements inverse iteration with shifts
'''

__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import scipy.linalg as la


def inverse_shift(A, sigma, n_iterations=15, x_0=np.array([0, 0, 1])):
    '''
        ARGS:
            A input matrix
            sigma the shift
            n_iterations self explanatory
            x_0 initial guess
        RETURNS:
            eigenvalue of A closest to sigma
            corresponding eigenvector 
    '''
    y = x_0  # present guess 
    n = A.shape[0]
    A_p = A - sigma * np.eye(n)
    lu_and_piv = la.lu_factor(A_p)
    eps_m = np.finfo(float).eps
    for k in range(0, n_iterations):
        v = y / np.linalg.norm(y)
        y = la.lu_solve(lu_and_piv, v)
        theta = np.dot(v, y)
        if (np.linalg.norm(y - theta * v) < theta * eps_m):
            break
    return sigma + 1.0 / theta, y / theta

A = np.array([[6, 2, 1],
              [2, 3, 1],
              [1, 1, 1]])
shift = 2
eigval, eigvec = inverse_shift(A, shift)
npeigvals, npeigvecs = np.linalg.eig(A)
mindiff = None
mindiff_loc = -1
for i in range(0, len(npeigvals)):
    if (mindiff is None):
        mindiff = npeigvals[i]
        mindiff_loc = i
        continue
    elif (np.abs(npeigvals[i] - shift) < mindiff):
        mindiff = np.abs(npeigvals[i] - shift)
        mindiff_loc = i

diffval = np.abs(npeigvals[mindiff_loc] - eigval)/npeigvals[mindiff_loc]
print(diffval)
print(eigvec)
print(npeigvecs[mindiff_loc])
