'''
    Perform:
        \integral_0^1 \frac{4}{1 + x^2}
    to approximate \pi

    Do this using:
        composite midpoint
        composite trapezoid
        composite simpsons

        + Romberg

        gauss-legendre
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import matplotlib.pyplot as plt

h = 10**-3
n = int(1.0 / h)
ns = range(100, 1000, 10)
xs_uniform = np.linspace(0, 1, n + 1)

pi = np.pi

lower = 0.0
upper = 1.0
interval = (lower, upper)


def f(x):
    return 4.0 / (1.0 + x**2)


def composite_midpoint(f, interval, n):
    '''
        Implements composite midpoint rule
        Integrates f over interval using n points
        ARGS
        -----
        f : function that takes in a scalar (x)
        interval: pair (a, b)
        n: int, number of panels to use for rule.
        RETURNS
        -----
        Qn : The approximate integral.
        count: number of function evaluations
    '''
    a, b = interval
    Qn = 0.0
    h = (b - a) / n
    ck = a + 0.5 * h
    count = 0
    while ck < b:
        Qn += h * f(ck)
        ck += h
        count += 1
    return Qn, count


def composite_trapezoidal(f, interval, n):
    '''
        Implements composite trapezoidal rule
        Integrates f over interval using n points
        ARGS
        -----
        f : function that takes in a scalar (x)
        interval: pair (a, b)
        n: int, number of panels to use for rule.
        RETURNS
        -----
        Qn : The approximate integral.
        count: number of function evaluations
    '''
    Qn = 0.0
    a, b = interval
    xis = np.linspace(a, b, n)
    h = (b - a) / n
    Qn += 0.5 * (f(a) + f(b))
    count = 2
    for i in range(1, len(xis) - 1):
        Qn += f(xis[i])
        count += 1
    # print('count = ', count)
    Qn *= h
    return Qn, count


def composite_simpson(f, interval, n):
    '''
        Implements composite simpson rule
        Integrates f over interval using n points
        ARGS
        -----
        f : function that takes in a scalar (x)
        interval: pair (a, b)
        n: int, number of panels to use for rule.
        RETURNS
        -----
        Qn : The approximate integral.
        count: number of function evaluations
    '''
    Qn = 0.0
    a, b = interval
    xis = np.linspace(a, b, n)
    h = (b - a) / n
    Qn += f(a) + f(b)
    count = 2
    for i in range(1, len(xis) - 1):
        Qn += f(xis[i]) * (2.0 if i % 2 == 0 else 4.0)
        count += 1
    # print('count = ', count)
    Qn *= (h / 3.0)
    return Qn, count


def gauss_legendre(f, interval, n):
    '''
        Implements gauss legendre quadrature rule
        Uses numpy to generate necessary points and weights
        Integrates f over interval using n points
        ARGS
        -----
        f : function that takes in a scalar (x)
        interval: pair (a, b)
        n: int, number of panels to use for rule.
        RETURNS
        -----
        Qn : The approximate integral.
        count: number of function evaluations
    '''
    a, b = interval
    xs_untransformed, ws = np.polynomial.legendre.leggauss(n)
    xs = list(map(lambda t: a + 0.5 * (t + 1.0) * (b - a), xs_untransformed))
    # Need to translate xs to a, b
    return 0.5 * (b - a) * np.dot(ws, np.array(list(map(f, xs)))), len(xs)


def romberg(f, interval, n0, nmax=10**6):
    # T[i][j] is:
    # apply trapezoidal on n0 * (2**i) points
    # apply richardson extrapolation j times
    # j <= i
    T = []
    n = n0
    i = 0
    T.append([composite_trapezoidal(f, interval, n)[0]])
    points = set()
    a, b = interval
    while n <= nmax:
        points = points | set(np.linspace(a, b, n))
        i += 1
        n = 2 * n
        T.append([])
        T[i].append(composite_trapezoidal(f, interval, n)[0])
        for j in range(1, i):
            c = 4**j
            T[i].append((c * T[i][j - 1] - T[i - 1][j - 1]) / (c - 1))
    return T[-1][-1], len(points)


# print('pi=', pi)
# print('n=', n)
# print('composite_midpoint=', composite_midpoint(f, interval, n))
# print('composite_trapezoidal=', composite_trapezoidal(f, interval, n))
# print('composite_simpson=', composite_simpson(f, interval, n))
# print('gauss_legendre=', gauss_legendre(f, interval, n))
# print('romberg=', romberg(f, interval, n // 8, nmax=n // 2))

cms = []
cts = []
css = []
gls = []
rms_counts = []
rms = []

for n in ns:
    cms.append(abs(composite_midpoint(f, interval, n)[0] - pi) / pi)
    cts.append(abs(composite_trapezoidal(f, interval, n)[0] - pi) / pi)
    css.append(abs(composite_simpson(f, interval, n)[0] - pi) / pi)
    gls.append(abs(gauss_legendre(f, interval, n)[0] - pi) / pi)

    rm, rmc = romberg(f, interval, n // 8, n // 2)
    rms.append(abs(rm - pi) / pi)
    rms_counts.append(rmc)

plt.loglog(ns, cms, 'r-^', label='composite_midpoint')
plt.loglog(ns, cts, 'g-*', label='composite_trapezoidal')
plt.loglog(ns, css, 'b-h', label='composite_simpson')
plt.loglog(ns, gls, 'k-s', label='gauss_legendre')
plt.loglog(rms_counts, rms, 'c-o', label='romberg')
plt.xlabel('log(# function calls)')
plt.ylabel('log(rel. error)')
plt.title('HW5 Q1: Relative error vs function calls')
plt.legend(loc='best')


print('Gauss legendre performs better than everything. The breaks in the graph must be points at which the error is 0')