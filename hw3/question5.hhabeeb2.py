#!/usr/bin/python
'''
    This file implements Fielder's algorithm.
    NOTE The author only implemented lanczos algorithm.
'''

__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pdb

def readmesh(fname):
    """
    Read a mesh file and return vertics as a (npts, 2)
    numpy array and triangles as (ntriangles, 3) numpy
    array. `npts` is the number of vertices of the mesh
    and `ntriangles` is the number of triangles of the
    mesh.
    """

    with open(fname, "r") as f:
        npoints = int(next(f))
        points = np.zeros((npoints, 2))
        for i in range(npoints):
            points[i, :] = [float(x) for x in next(f).split()]

        ntriangles = int(next(f))
        triangles = np.zeros((ntriangles, 3), dtype=int)
        for i in range(ntriangles):
            triangles[i, :] = [int(x) - 1 for x in next(f).split()]

    return points, triangles


def plotmesh(points, triangles, tricolors=None):
    """
    Given a list of points (shape: (npts, 2)) and triangles
    (shape: (ntriangles, 3)), plot the mesh.
    """

    plt.figure()
    plt.gca().set_aspect('equal')
    if tricolors is None:
        plt.triplot(points[:, 0], points[:, 1], triangles, 'bo-', lw=1.0)
    else:
        plt.tripcolor(points[:, 0],
                      points[:, 1],
                      triangles,
                      facecolors=tricolors,
                      edgecolors='k')
    plt.show()
    return


def mesh2dualgraph(triangles):
    """
    Calculate the graph laplacian of the dual graph associated
    with the mesh given by numpy array traingles.
    """

    n, m = triangles.shape

    assert(m == 3), "Triangle should have exactly three points !!"

    G = np.zeros((n, n))

    for i, ti in enumerate(triangles):
        for j, tj in enumerate(triangles):
            # If there is a common edge
            if len(set(ti) - set(tj)) == 1:
                G[i, j] = G[j, i] = -1

    for i in range(n):
        G[i, i] = -np.sum(G[i, :])

    return ss.csr_matrix(G)


def lanczos(A, x0, iterations, thr=np.finfo(float).eps):
    '''
        A: laplacian of dual graph... any matrix, really?
        x0: starting point
        iterations: number of iterations.
    '''
    n = A.shape[0]
    Q = np.zeros((n, 1 + iterations), dtype=np.float)
    # T = np.zeros((iterations, iterations))
    Q[:, 0] = np.zeros((n,), dtype=np.float)
    Q[:, 1] = x0 / np.linalg.norm(x0)
    alpha = []
    beta = [0.0]
    for k in range(1, iterations):
        u = A * Q[:, k]
        u0 = np.linalg.norm(u)
        alpha.append(np.dot(Q[:, k], u))
        u = u - beta[-1] * Q[:, k - 1] - alpha[-1] * Q[:, k]
        beta.append(np.linalg.norm(u))
        if (beta[-1] / u0 < thr):
            break
        Q[:, k + 1] = u / beta[-1]
    pdb.set_trace()
    T = np.zeros((iterations, iterations))
    for i in range(0, len(alpha)):
        T[i, i] = alpha[i]
        if (i + 1 < iterations):
            T[i + 1, i] = beta[i + 1]
            T[i, i + 1] = beta[i + 1]
    return Q[:, 1:], T

# def lanczos(A, x0, iterations, thr=np.finfo(float).eps):
#     '''
#         A: laplacian of dual graph... any matrix, really?
#         x0: starting point
#         iterations: number of iterations.
#     '''
#     n = A.shape[0]
#     Q = np.zeros((n, 1 + iterations), dtype=np.float)
#     q = x0 / np.linalg.norm(x0)
#     Q[:, 1] = q
#     r = A * q
#     alpha = [np.dot(q, r)]
#     r = r - alpha[-1] * q
#     beta = [np.linalg.norm(r)]
#     for j in range(2, iterations):
#         v = q
#         q = r / beta[-1]
#         Q[:, j] = q
#         r = A * q - beta[-1] * v
#         alpha.append(np.dot(q, r))
#         r = r - alpha[-1] * q
#         beta.append(np.linalg.norm(r))
#         if (beta[-1] < thr):
#             break
#     pdb.set_trace()
#     T = np.zeros((iterations, iterations))
#     for i in range(0, len(alpha)):
#         T[i, i] = alpha[i]
#         if (i < (len(alpha) - 1)):
#             T[i, i + 1] = beta[i]
#             T[i + 1, i] = beta[i]
    # return Q[:, 1:], T

def fiedler(G, k):
    """
    Calculate the fiedler vector of the graph Laplacian matrix
    `G` using `k` iterations of Lanczos algorithm.
    """
    n, m = G.shape

    assert (n == m), "Matrix should be square !!"

    x0 = np.linspace(1, n, num=n)

    # You should complete this Lanczos function
    Q, T = lanczos(G, x0, k)

    eVal, eVec = np.linalg.eig(T)
    idx = eVal.argsort()
    eVal = eVal[idx]
    eVec = eVec[:, idx]
    fiedlerVec = np.dot(Q, eVec[:, 1])

    partitionVec = np.zeros_like(fiedlerVec)
    mfeidler = np.ma.median(fiedlerVec)

    for i in range(n):
        if fiedlerVec[i] >= mfeidler:
            partitionVec[i] = 1
        else:
            partitionVec[i] = -1

    return partitionVec


points, triangles = readmesh("mesh.1")
plotmesh(points, triangles)
G = mesh2dualgraph(triangles)
partitionVec = fiedler(G, 150)
plotmesh(points, triangles, partitionVec)
