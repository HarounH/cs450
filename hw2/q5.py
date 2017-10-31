#!/usr/bin/python
'''HW2.Q5 for CS450 FAll 2017.'''
__author__ = 'haroun habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import pdb
import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt


def generate_hilbert(n):
    hilbert = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            hilbert[i, j] = 1.0 / (i + j + 1)  # One indexing vs 0 indexing
    return hilbert


def gs(H):
    # Performs QR decomposition
    # Using Gram-Schmidt
    # Takes in a matrix as input
    # Returns Q
    n = H.shape[0]
    Q = np.eye(n)
    R = np.eye(n)
    for k in range(0, n):
        Q[:, k] = H[:, k].copy()
        for j in range(0, k):
            R[k, j] = R[j, k] = np.dot(Q[:, j], H[:, k])
            Q[:, k] -= R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]
    return Q


def mgs(H):
    # Performs QR decomposition
    # Using Modified Gram-Schmidt
    # Takes in a matrix as input
    # Returns Q
    n = H.shape[0]
    Q = np.eye(n)
    for k in range(0, n):
        Q[:, k] = H[:, k] / np.linalg.norm(H[:, k])
        for j in range(k + 1, n):
            H[:, j] -= np.dot(Q[:, k], H[:, j]) * Q[:, k]
    return Q


def sgn(x):
    return (-1.0 if x < 0.0 else 1.0)

def householder(A):
    # Performs QR decomposition 
    # Using householder
    # Takes in a matrix as input
    # Returns Q
    n = A.shape[0]
    eye = np.eye(n)
    Q = np.eye(n)
    for i in range(0, n):
        ai = A[i:, i]  # Only need to consider a submatrix
        ei = eye[i:, i]
        v = ai + sgn(ai[0]) * ei * np.linalg.norm(ai)
        matv = np.array([v.tolist()])
        Hi = np.eye(n - i) - (2 * np.transpose(matv) @ matv) / np.dot(v, v)
        Hi = scla.block_diag(np.eye(i), Hi)
        A = Hi @ A
        Q = Hi @ Q
    return Q


def accuracy(Q):
    return -np.log10(np.linalg.norm(np.eye(Q.shape[0]) - np.transpose(Q) @ Q))


hilbert = []
ns = list(range(2, 13))
for n in ns:
    hilbert.append(generate_hilbert(n))

gs_q = [gs(h.copy()) for h in hilbert]
gs_qq = [gs(q.copy()) for q in gs_q]
mgs_q = [mgs(h.copy()) for h in hilbert]
hs_q = [householder(h.copy()) for h in hilbert]

gs_q_acc = [accuracy(q) for q in gs_q]
gs_qq_acc = [accuracy(q) for q in gs_qq]
mgs_q_acc = [accuracy(q) for q in mgs_q]
hs_q_acc = [accuracy(q) for q in hs_q]


plt.figure(1)
plt.plot(ns, gs_q_acc, 'b-o', label='Gram-Schmidt')
plt.plot(ns, gs_qq_acc, 'r--D', label='Gram-Schmidt of Gram-Schmidt')
plt.plot(ns, mgs_q_acc, 'g-.s', label='Modified Gram-Schmidt')
plt.plot(ns, hs_q_acc, 'c:v', label='Householder')
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('n')
plt.ylabel('digits')
plt.legend(loc='best')
plt.title('digits of accuracy vs n : hhabeeb2.HW2.Q5')
plt.show()
