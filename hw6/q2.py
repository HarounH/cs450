'''
    Solving HW6 Q2 for CS450 Fall 2017
    Eigenvalue problems for smallest eigenvalues

    Part 1: Dirichlet problem
        -u'' = \lambda u; u(0) = u(1) = 0

        Actual solution: u(x) = sink\pi x, \lambda_k = k^2\pi^2

    Part 2: Neumann problem
        -u'' = \lambda u; u(0) = u'(1) = 0

        Actual solution:
            u(x) = sin(k - \frac{1}{2})\pi x,
            \lambda_k = {k - \frac{1}{2}}^2\pi^2
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import pdb
import numpy as np
import scipy.sparse as sp
from scipy.optimize import broyden1, newton_krylov
import matplotlib.pyplot as plt

dltrue = np.pi ** 2
nltrue = np.pi ** 2 / 4


def dutrue(n):
    return np.sin(np.pi * np.linspace(0, 1, n + 2)[1:-1])


def nutrue(n):
    return np.sin(np.pi * 0.5 * np.linspace(0, 1, n + 2)[1:])


def solve(ns):
    '''
        -u'' = l * u

    '''
    dle = []
    due = []
    nle = []
    nue = []
    for n in ns:
        dut = dutrue(n)
        nut = nutrue(n)
        # dlt = np.pi**2
        # Dirichlet
        A = sp.diags([-n - 1, 2 * (n + 1), -(n + 1)],
                     [-1, 0, 1],
                     shape=(n, n),
                     format='csr')
        B = sp.diags([1 / (6 * (n + 1)), 2 / (3 * (n + 1)), 1 / (6 * (n + 1))],
                     [-1, 0, 1],
                     shape=(n, n),
                     format='csr')
        ls, us = sp.linalg.eigs(A, M=B, k=1, which='SM')  # smallest magnitude
        us = us.reshape(-1)
        # pdb.set_trace()
        dle.append(abs(ls[0] - dltrue))
        due.append(np.linalg.norm(us - dut, ord=np.inf))  # /
                   # np.linalg.norm(dut))
        A = sp.diags([-n - 1, 2 * (n + 1), -(n + 1)],
                     [-1, 0, 1],
                     shape=(n + 1, n + 1),
                     format='csr')
        B = sp.diags([1 / (6 * (n + 1)), 2 / (3 * (n + 1)), 1 / (6 * (n + 1))],
                     [-1, 0, 1],
                     shape=(n + 1, n + 1),
                     format='csr')

        ls, us = sp.linalg.eigs(A, M=B, k=1, which='SM')  # smallest magnitude
        us = us.reshape(-1)[:-1]
        pdb.set_trace()
        nle.append(abs(ls[0] - nltrue))
        nue.append(np.linalg.norm(us - nut, ord=np.inf))  # /
    pdb.set_trace()
    plt.loglog(ns, dle, 'r--', label='lambda dirichlet error')
    plt.loglog(ns, due, 'r-', label='u dirichlet error')
    plt.loglog(ns, nle, 'r--', label='lambda neuman error')
    plt.loglog(ns, nue, 'r-', label='u neuman error')
    plt.loglog(ns, [float(n)**-2 for n in ns], 'k--', label='O(n^-2)')
    plt.xlabel('log(n)')
    plt.ylabel('log(error)')
    plt.title('error vs n')
    plt.legend(loc='best')
    plt.show()


def solve2(ns):
    dle = []
    due = []
    nle = []
    nue = []
    for n in ns:
        dut = dutrue(n)
        nut = nutrue(n)
        # Dirichlet
        A = sp.diags([-n -1, 2.0*(n + 1), -n -1],
                     [-1, 0, 1],
                     shape=(n, n),
                     format='csr')
        B = sp.diags([1 / (6 * (n + 1)), 2 / (3 * (n + 1)), 1 / (6 * (n + 1))],
                     [-1, 0, 1],
                     shape=(n, n),
                     format='csr')

        ls, us = sp.linalg.eigs(A, M=B, k=1, which='SM')
        dl = ls[0]
        dus = us.reshape(-1)
        dle.append(abs(ls[0] - dltrue))
        due.append(np.linalg.norm(dus - dut) / np.linalg.norm(dut))
        due[-1] = np.linalg.norm(A @ dus - dl * B @ dus, ord=np.inf)
        # Neumann
        A = sp.diags([-n -1, 2.0*(n + 1), -n -1],
                     [-1, 0, 1],
                     shape=(n + 1, n + 1),
                     format='csr')
        A[-1, -1] *= 0.5  # Only half the \phi is inside [0, 1]
        B = sp.diags([1 / (6 * (n + 1)), 2 / (3 * (n + 1)), 1 / (6 * (n + 1))],
                     [-1, 0, 1],
                     shape=(n + 1, n + 1),
                     format='csr')
        B[-1, -1] *= 0.5

        ls, us = sp.linalg.eigs(A, M=B, k=1, which='SM')
        nus = us.reshape(-1)
        nl = ls[0]
        nle.append(abs(nl - nltrue))
        nue.append(np.linalg.norm(nus - nut) / np.linalg.norm(nut))
        nue[-1] = np.linalg.norm(A @ nus - nl * B @ nus, ord=np.inf)
    plt.loglog(ns, dle, 'r--', label='lambda dirichlet error')
    # plt.loglog(ns, due, 'r-', label='u dirichlet error')
    plt.loglog(ns, nle, 'r--', label='lambda neuman error')
    # plt.loglog(ns, nue, 'r-', label='u neuman error')
    plt.loglog(ns, [float(n)**-2 for n in ns], 'k--', label='O(n^-2)')
    plt.xlabel('log(n)')
    plt.ylabel('log(error)')
    plt.title('error vs n')
    plt.legend(loc='best')
    plt.show()

    print('Going from dirichlet to neumann is achieved by including an extra \pi_i. However, this function is centered at 1. The form of the linear equation remains the same - Ax=\lambda Bx. However, the dimensionality of A, B has increased by 1 to account for \phi_{n + 1}. Another important modification that is made is to change the values of a_{n+1}{n+1} and b_{n+1}{n+1}. They are half of a_{i}{i}, b_{i}{i} respectively. Again, this is because only half of \phi_{n+1} is in the integration limits. Also, the step size (h) is the same for both cases.')


solve2(np.array([2**i for i in range(2, 10)]))
