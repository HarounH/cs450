'''
    Solves cp.11.13 of the scientific computing book by michael heath.
    Implements gauss siedel and jacbo iteration schemes
    tests for finite difference approximation to 1d laplace equation

    Note: still solve Ax=b, lol
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

N_MAX_ITER = 10
EPS = 10**-6


def gs(A, b, x0, max_iter=N_MAX_ITER, eps=EPS):
    '''
        Solves the linear system
            Ax = b
        for x using the gauss siedel method
    '''
    xs = [x0]
    rs = [b - A @ x0]
    niter = 0
    D = np.diag(np.diag(A))  # Ah, silly numpy.
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    M = D + L
    while niter < max_iter and np.linalg.norm(rs[-1]) > eps:
        xs.append(np.linalg.solve(M, b - U @ xs[-1]))
        rs.append(b - A @ xs[-1])
        niter += 1
    return xs[-1], xs, rs


def jacobi(A, b, x0, max_iter=N_MAX_ITER, eps=EPS):
    '''
        Solves the linear system
            Ax = b
        for x using the jacobi method
    '''
    xs = [x0]
    rs = [b - A @ x0]
    niter = 0
    D = np.diag(np.diag(A))  # Ah, silly numpy.
    # pdb.set_trace()
    Dinv = np.linalg.inv(D)
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    M = L + U
    while niter < max_iter and np.linalg.norm(rs[-1]) > eps:
        xs.append(Dinv @ (b - M @ xs[-1]))
        rs.append(b - A @ xs[-1])
        niter += 1
    return xs[-1], xs, rs


if __name__ == '__main__':
    n = 50
    js = np.array(range(1, n + 1))

    ks = [1] + list(range(5, 51, 5))
    lineparams = ['r-',
                  'b-',
                  'c-',
                  'g-',
                  'k-',
                  'm-',
                  'r--',
                  'b--',
                  'c--',
                  'g--',
                  'k--',
                  'm--']
    b = np.zeros((n,))
    A = sp.diags([2 * np.ones((n,)),
                  -1 * np.ones((n - 1,)),
                  -1 * np.ones((n - 1,))],
                 [0, 1, -1]).toarray()
    gs_errors = []
    jacobi_errors = []
    for k in ks:
        x0 = np.sin(js * np.pi * k / (n + 1))
        _, _, gs_error = gs(A, b, x0)
        gs_errors.append(gs_error)
        _, _, jacobi_error = jacobi(A, b, x0)
        jacobi_errors.append(jacobi_error)
    # Plot error vs iteration
    fig = plt.figure(1)
    for i in range(len(ks)):
        # pdb.set_trace()
        k = ks[i]
        plt.plot(range(len(gs_errors[i])),
                 [np.linalg.norm(e) for e in gs_errors[i]],
                 lineparams[i],
                 label='GS k=' + str(k))
    plt.legend(loc='best')
    plt.title('Gauss Siedel')
    plt.xlabel('iteration')
    plt.ylabel('||error||')
    plt.savefig('../cp.11.13.gauss-siedel.png')

    fig = plt.figure(2)
    for i in range(len(ks)):
        k = ks[i]
        plt.plot(range(len(jacobi_errors[i])),
                 [np.linalg.norm(e) for e in jacobi_errors[i]],
                 lineparams[i],
                 label='jacobi k=' + str(k))
    plt.legend(loc='best')
    plt.title('Jacobi')
    plt.xlabel('iteration')
    plt.ylabel('||error||')
    plt.savefig('../cp.11.13.jacobi.png')
    pdb.set_trace()
    # fig = plt.figure(3)
    # plt.subplot(231)
    # for k <= 25
    # for j in range(len(gs_errors[0]))
    # plt.plot(range(len(gs_errors[0])), gs_errors[0])
    # fig = plt.figure(4)
    # plt.subplot(231)
    # for k >= 25
    plt.show()
