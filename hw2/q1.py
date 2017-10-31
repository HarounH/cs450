#!/usr/bin/python
'''Solution for hw2.q1 of CS450 Fall 2017'''

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import time
import pdb
import matplotlib.pyplot as plt
# Returns the product (A x B x C)u
def kron_eval(A, B, C, u):
    ma, na = A.shape
    mb, nb = B.shape
    mc, nc = C.shape

    v = np.zeros((nc, mb, na))
    u = u.reshape((nc, nb, na))

    Bt = B.transpose()

    for k in range(na):
        v[:, :, k] = u[:, :, k] @ Bt

    v = v.reshape((nc, nb * na))
    v = C @ (v)

    v = v.reshape((mc * nb, nc))
    v = v @ (A.transpose())

    v = v.reshape((ma * mb * mc, 1))
    return v



def main(Ns):
    '''Ns: Array like containing integers'''
    init_times = []
    spsolve_times = []
    fastsolve_times = []

    for N in Ns:
        # create Ax
        tic = time.time()

        temp = (N + 1) ** 2
        h = 1.0 / (N + 1)
        diagonals = np.array([[-temp] * (N - 1), [2 * temp] * N, [-temp] * (N - 1)])
        # diagonals *= (N + 1)**2  # why even .... I HATE THIS
        offsets = [-1, 0, 1]
        Ax = sp.diags(diagonals, offsets)

        # generate A as 3 parts and then add
        A1 = sp.kron(sp.identity(N), sp.kron(sp.identity(N), Ax))
        A2 = sp.kron(sp.identity(N), sp.kron(Ax, sp.identity(N)))
        A3 = sp.kron(Ax, sp.kron(sp.identity(N), sp.identity(N)))
        A = A1 + A2 + A3  # irrelevant

        toc = time.time()
        init_times.append(toc - tic)
        print('Initialization of Ax, A, f for N=', N, ' took ', toc - tic)

        f = np.random.rand(N**3, 1)

        # Sparse solver
        tic = time.time()
        u_sp = la.spsolve(A, f)
        toc = time.time()
        spsolve_times.append(toc - tic)
        print('Sparse solver for N=', N, ' took ', toc - tic)

        # Fast solver
        tic = time.time()
        u_fs = fast_solve(Ax, f)
        toc = time.time()
        fastsolve_times.append(toc - tic)
        print('Solving (+factorization) for N=', N, ' took ', toc - tic)
        
    return init_times, spsolve_times, fastsolve_times


def fast_solve(Ax, f):
    '''
        Returns u | Ax @ u ~= f
    '''
    N = Ax.shape[0]  # Number of rows
    h = 1.0 / (N + 1)

    # Factorization
    Lambda_x = sp.diags([[(1.0 - np.cos(np.pi * h * (i + 1))) * 2.0 / (h**2)
                          for i in range(0, N)]], [0])
    sqrt2h = np.sqrt(2 * h)
    S_x = np.array([[sqrt2h * np.sin(np.pi * h * (i + 1) * (j + 1)) for j in range(0, N)]
                    for i in range(0, N)])
    S_x_inv = np.transpose(S_x)
    # solve
    u0 = kron_eval(S_x_inv, S_x_inv, S_x_inv, f)
    D = sp.kron(sp.identity(N), sp.kron(sp.identity(N), Lambda_x)) \
        + sp.kron(sp.identity(N), sp.kron(Lambda_x, sp.identity(N))) \
        + sp.kron(Lambda_x, sp.kron(sp.identity(N), sp.identity(N)))
    Dinv = sp.diags([[1.0 / x for x in D.diagonal()]], [0])
    u1 = Dinv @ u0
    u2 = kron_eval(S_x, S_x, S_x, u1)

    return u2


# Just in case we have no if __name__=='__main__':
Ns = list(range(3, 20))
init_times, spsolve_times, fastsolve_times = main(Ns)
# pdb.set_trace()
plt.figure(1)
plt.plot(Ns, spsolve_times, 'b-', label='Sparse Solver')
plt.plot(Ns, fastsolve_times, 'r--', label='Fast Solver')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('time')
plt.legend(loc='best')
plt.title('hhabeeb2.HW2.Q1')
plt.show()
