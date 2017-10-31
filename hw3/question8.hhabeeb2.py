#!/usr/bin/python
'''
    This file implements
        Newton's method
        Broyden's method
    in HW3 of CS450 Fall 2017
'''

__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import scipy.linalg as la
import pdb
import matplotlib.pyplot as plt

def newton(m, n, f, J, x0, thr=np.finfo(float).eps):
    '''
        Finds root of f using broyden's method
        Does not perform rank-one updates of J
        Solves for update instead
        ARGS:
            m int # of scalar functions
            n int dimensionality of input
            f list of m R^n -> R functions
            x0 R^n (float list) initial guess
            thr float threshold for accuracy.
        RETURNS
            xs list of R^n representing guesses at each iteration
    '''
    xs = [x0]
    while True:
        sk = la.solve(J(xs[-1]), -f(xs[-1]))
        xs.append(xs[-1] + sk)
        # print('newton:', len(xs), ' e=', np.abs(f(xs[-1])), end='\n')
        if ((np.abs(f(xs[-1])) < thr).all()):
            break
    return xs


def broyden(m, n, f, x0, B0, thr=np.finfo(float).eps):
    '''
        Finds root of f using broyden's method
        Does not perform rank-one updates of J
        Solves for update instead
        ARGS:
            m int # of scalar functions
            n int dimensionality of input
            f list of m R^n -> R functions
            x0 R^n (float list) initial guess
            B0 m x n matrix, initial jacobian.
            thr float threshold for accuracy.
        RETURNS
            xs list of R^n representing guesses at each iteration
    '''

    xs = [x0]
    Bs = [B0]
    while True:   
        # print('broyden:', len(xs), end='\r')
        sk = la.solve(Bs[-1], -f(xs[-1]))  # We're allowed to solve every itreration
        xs.append(xs[-1] + sk)
        y = f(xs[-1]) - f(xs[-2])
        Bs.append(Bs[-1] + (np.outer((y - (Bs[-1] @ sk)), np.transpose(sk)) / np.dot(sk, sk)))
        if ((np.abs(f(xs[-1])) < thr).all()):
            break
    return xs


def f1(x):
    '''
        ARGS
            x: R^2
        RETURNS:
            ans float
    '''
    return 18.0 + (x[0] + 3.0) * (x[1]**3 - 7.0)


def f2(x):
    '''
        ARGS
            x: R^2
        RETURNS:
            ans float
    '''
    return np.sin((x[1] * (np.e**x[0])) - 1.0)


def jacobian(x):
    '''
        ARGS:
            x: R^2
        RETURNS:
            jacobian of [f1, f2]
    '''
    return np.array([[(x[1]**3 - 7), 3 * (x[0] + 3) * (x[1]**2)],
                     [np.cos((x[1] * (np.e**x[0])) - 1.0) * x[1] * (np.e**x[0]),
                      np.cos((x[1] * (np.e**x[0])) - 1.0) * (np.e**x[0])
                      ]])


def f(x):
    return np.array([f1(x), f2(x)])


J = jacobian
x0 = np.array([-0.5, 1.4])
xtrue = np.array([0, 1])
# print('starting newton')
newtonxs = newton(2, 2, f, J, x0)
# print('done with newton.\nstarting broyden')
broydenxs = broyden(2, 2, f, x0, J(x0))
# print('done with broyden')

newton_e = list(map(lambda x: np.linalg.norm(x - xtrue), newtonxs))
broyden_e = list(map(lambda x: np.linalg.norm(x - xtrue), broydenxs))

plt.semilogy(list(range(0, len(newtonxs))), newton_e, 'r', label='newton')
plt.semilogy(list(range(0, len(broydenxs))), broyden_e, 'b', label='broyden')
plt.xlabel('iteration')
plt.ylabel('log(error)')
plt.title('log(error) vs iteration')
plt.legend(loc='best')
plt.show()

print('newton:', len(newtonxs))
print('broyden:', len(broydenxs))
