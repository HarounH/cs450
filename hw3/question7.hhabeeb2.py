#!/usr/bin/python
'''
    This file implements fxd pnt iterations for Q7 in HW3 of CS450 Fall 2017
'''

__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import matplotlib.pyplot as plt
import pdb


def run(x_0, g, n_iterations=10):
    '''
        Runs x <- g(x) for n_iterations
        with x_0 as the starting point
        ARGS:
            x_0 is any input of type T
            g is a function of type T->T
            n_iterations int
        RETURNS:
            list of T representing sequence
    '''
    ans = [x_0]
    for i in range(0, n_iterations):
        ans.append(g(ans[-1]))
    return ans


def g1(x):
    return (2.0 + x**2) / 3.0


def g2(x):
    return np.sqrt(3 * x - 2)


def g3(x):
    return 3.0 - (2.0 / x)


def g4(x):
    return ((x**2) - 2) / ((2 * x) - 3)


x_0 = 5.0
n_iterations = 10
iters = list(range(0, n_iterations + 1))
e1s = list(map(lambda x: np.abs((x - 2) / 2.0), run(x_0, g1, n_iterations)))
e2s = list(map(lambda x: np.abs((x - 2) / 2.0), run(x_0, g2, n_iterations)))
e3s = list(map(lambda x: np.abs((x - 2) / 2.0), run(x_0, g3, n_iterations)))
e4s = list(map(lambda x: np.abs((x - 2) / 2.0), run(x_0, g4, n_iterations)))

plt.semilogy(iters, e1s, 'r', label='g1')
plt.semilogy(iters, e2s, 'b', label='g2')
plt.semilogy(iters, e3s, 'y', label='g3')
plt.semilogy(iters, e4s, 'g', label='g4')
plt.legend(loc='best')
plt.title('All : log(error) vs iteration')
plt.ylabel('log(error)')
plt.xlabel('iteration')
plt.show()


# plt.semilogy(iters, e1s, 'r', label='g1')
plt.semilogy(iters, e2s, 'b', label='g2')
plt.semilogy(iters, e3s, 'y', label='g3')
plt.semilogy(iters, e4s, 'g', label='g4')
plt.legend(loc='best')
plt.title('Convergent : log(error) vs iteration')
plt.ylabel('log(error)')
plt.xlabel('iteration')
plt.show()
