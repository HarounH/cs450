'''
    Implement steepest descent and conj gradient method to solve
    \(Ax = b\)
    where A is positive definite and symmetric.

    So basically, optimize xAx / 2 - xb
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'


import pdb
import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
from matrix_generator import generate_matrix, condition_matrix
from matrix_generator import define_poisson_system, define_heat
MAX_ITER = 100


def cg(A, b, x0, max_iter=MAX_ITER, eps=10**-6):
    '''
        minimizes 0.5 * x^T A x - x^T b,
        which is equivalent to solving Ax = b

        Does so using Conjugate gradient descent.

        ARGS
        ---------
        A : n*n symmetric positive definite matrix
        b : Ax = b
        x0: array of shape (n,) initial guess for x0
        max_iter: int, maximum number of iterations to run
        eps: float, threshold on residual for stopping
        RETURNS
        ---------
        xs[-1]: The approximate solution to Ax = b
        xs : sequence of steps
        rs : sequence of residuals
    '''
    # pdb.set_trace()
    tic = time.time()
    xs = [x0]
    r_old = b - A @ x0
    ss = [r_old]
    rs = [r_old]
    niter = 0
    while niter < max_iter and np.linalg.norm(rs[-1]) > eps:
        # niter iterations have been completed
        r_old_norm = r_old.T @ r_old
        sk_A_norm = ss[-1].T @ A @ ss[-1]
        ak = r_old_norm / sk_A_norm
        xs.append(xs[-1] + ak * ss[-1])
        r_new = r_old - ak * A @ ss[-1]
        bkp1 = r_new.T @ r_new / r_old_norm
        ss.append(r_new + bkp1 * ss[-1])
        r_old = r_new
        rs.append(r_new)
        niter += 1
    toc = time.time()
    # pdb.set_trace()
    return xs[-1], xs, rs, toc - tic


def sd(A, b, x0, max_iter=MAX_ITER, eps=10**-10):
    '''
        minimizes 0.5 * x^T A x - x^T b,
        which is equivalent to solving Ax = b

        Does so using steepest descent.

        ARGS
        ---------
        A : n*n symmetric positive definite matrix
        b : Ax = b
        x0: array of shape (n,) initial guess for x0
        max_iter: int, maximum number of iterations to run
        eps: float, threshold on residual for stopping
        RETURNS
        ---------
        xs[-1]: The approximate solution to Ax = b
        xs: sequence of x
        rs: sequence of residuals
    '''
    tic = time.time()
    xs = [x0]
    rs = [b - A @ xs[-1]]
    niter = 0
    while niter < max_iter and np.linalg.norm(rs[-1]) > eps:
        r = rs[-1]
        ak = (r.T @ r) / (r.T @ A @ r)
        xs.append(xs[-1] + ak * r)
        rs.append(b - A @ xs[-1])
        niter += 1
    toc = time.time()
    return xs[-1], xs, rs, toc - tic


class Data:
    def __init__(self):
        pass


if __name__ == '__main__':
    As = [np.array([[2, -1, 0],
                    [-1, 2, -1],
                    [0, -1, 2]])]
    bs = [np.array([2, 3, 1], dtype=float)]
    x0s = [np.array([-3, 4, 1], dtype=float)]
    names = ['sanity']

    # Well conditioned
    As.append(condition_matrix(generate_matrix(4), 3.0))
    bs.append(4 * np.random.rand(4))
    x0s.append(np.random.rand(4))
    names.append('4x4-well-conditioned')

    # ill conditioned
    As.append(condition_matrix(generate_matrix(4), 10**7))
    bs.append(4 * np.random.rand(4))
    x0s.append(np.random.rand(4))
    names.append('4x4-ill-conditioned')


    # large system
    names.append('poisson.n20')
    define_poisson_system(As, bs, x0s, n=20)

    # names.append('poisson.n50')
    # define_poisson_system(As, bs, x0s, n=50)

    names.append('heat.a0.1.n   20')
    define_heat(As, bs, x0s, a=0.1, n=20)

    for i in range(0, len(As)):
        A = As[i]
        b = bs[i]
        x0 = x0s[i]
        name = names[i]
        cg_data = Data()
        sd_data = Data()
        cg_data.x, cg_data.xs, cg_data.rs, cg_data.time = cg(A, b, x0)
        sd_data.x, sd_data.xs, sd_data.rs, sd_data.time = sd(A, b, x0)
        # pdb.set_trace()
        # Plot stuff
        fig = plt.figure(i)

        # plots
        plt.semilogy(list(range(0, len(cg_data.xs))),
                 [np.linalg.norm(r) for r in cg_data.rs],
                 'r-',
                 label='conjugate: ' + str(cg_data.time))
        plt.semilogy(list(range(0, len(sd_data.xs))),
                 [np.linalg.norm(r) for r in sd_data.rs],
                 'b-',
                 label='steepest: ' + str(sd_data.time))
        plt.title(name)
        plt.ylabel('log(norm(r))')
        plt.xlabel('iteration')
        plt.legend(loc='best')
        plt.savefig('../cp.11.12.' + name + '.png')
    # plt.show()
