__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import matplotlib.pyplot as plt


def f(x, k=15.0):
    return np.cos(k * x)


def deltaf(x, k=15.0):
    return -k * np.sin(k * x)


def derivate_matrix(xs):
    '''
        computes derivative matrix as specified in class notes
            d_ij = c_j / (c_i * (x_i - x_j)) \forall i \neq j
            d_ii = \Sigma_{k \neq i} \frac{1}{x_i - x_k}
        ARGS
        -------
        xs: masked array of x, with mask=False

        RETURNS
        -------
        D: the derivative matrix with d_ij as specificed above
    '''
    cinvs = np.zeros(n)
    for i in range(0, n):
        xi = xs[i]
        xs.mask[i] = True
        cinvs[i] = (xi - xs).prod()
        xs.mask[i] = False
    # import pdb; pdb.set_trace()
    D = np.zeros((n, n), dtype=float)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                xi = xs[i]
                xs.mask[i] = True
                D[i, i] = ((xi - xs)**-1).sum()
                xs.mask[i] = False
            else:
                D[i, j] = cinvs[i] / (cinvs[j] * (xs[i] - xs[j]))
    return D


lower = -1.0
upper = 1.0
interval = (-1.0, 1.0)

ns = range(2, 51, 1)
uniform_errors = []
legendre_errors = []
for n in ns:
    xs_uniform = np.ma.array(np.linspace(lower, upper, n),
                             mask=False)
    xs_legendre = np.ma.array(np.polynomial.legendre.leggauss(n)[0],
                              mask=False)
    Dhat_uniform = derivate_matrix(xs_uniform)
    Dhat_legendre = derivate_matrix(xs_legendre)
    f_uniform = np.array(list(map(f, xs_uniform)))
    f_legendre = np.array(list(map(f, xs_legendre)))

    interp_gradient_uniform = Dhat_uniform @ f_uniform
    interp_gradient_legendre = Dhat_legendre @ f_legendre
    real_gradient_uniform = np.array(list(map(deltaf, xs_uniform)))
    real_gradient_legendre = np.array(list(map(deltaf, xs_legendre)))
    uniform_errors.append(abs(interp_gradient_uniform -
                              real_gradient_uniform).max())
    legendre_errors.append(abs(interp_gradient_legendre -
                               real_gradient_legendre).max())
    # import pdb; pdb.set_trace()

plt.semilogy(ns, uniform_errors, 'r-', label='uniform')
plt.semilogy(ns, legendre_errors, 'b-', label='legendre')
plt.xlabel('n')
plt.ylabel('log(error)')
plt.legend(loc='best')
plt.title('log(error) vs n')
# plt.show()

print('As expected, we see that the error when using legendre interpolation decreasing really quickly. We also see that for large n, uniformly spaced points perform poorly.')
