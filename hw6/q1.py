'''
    Solution to HW6 Q1 of CS450 Fall 2017

    Solving
        u'' = 10u**3 + 3u + t**2

    u(0) = u(1) = 1
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import pdb
import numpy as np
from scipy.optimize import broyden1
import matplotlib.pyplot as plt

EPS = 10**-14


def solve(n):
    h = 1.0 / (n + 1)
    starting_guess = np.array([i * h for i in range(0, n + 2)])
    '''
        Using finite difference,
        we have:
            (1/h**2)*(u_{i + 1} + u_{i - 1} - 2u_i) = 10u_i^3 + 3u_i + (ih)**2

        i = 1, 2, 3, ... n

            u_0 = 0
            u_{n + 1} = 0

        Hence, given x \in R^{n + 2}, we're finding the root to
        the above system of equations
    '''
    def f(x):
        '''
            x:
        '''
        ans = np.zeros(n + 2)
        ans[0] = x[0]
        ans[-1] = x[-1]
        for i in range(1, n + 1):
            ans[i] = ((n + 1)**2) * (x[i + 1] + x[i - 1] - 2 * x[i]) + \
                - 10 * (x[i]**3) - 3 * x[i] - (i * h)**2
        return ans
    sol = broyden1(f, starting_guess, f_tol=EPS)
    return sol


# Cant have ifmain because of submission portal quirks
ns = [1, 3, 7, 15]
lineargs = {1: 'r-o',
            3: 'b-o',
            7: 'k-s',
            15: 'c--v'}
n2u = {}
n2t = {}
for n in ns:
    n2u[n] = solve(n)
    n2t[n] = np.linspace(0, 1, n + 2)
    plt.plot(n2t[n], n2u[n], lineargs[n], label='n=' + str(n))
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend(loc='best')
plt.title('u(t) vs t')
plt.show()
