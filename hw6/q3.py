__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import pdb
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

xlo = -1.0
xhi = 2.0
tlo = 0.0
thi = 1.0
real_sigma = 400.0
real_c = 1.0


def utrue(lo, hi, n, t, sig=real_sigma, const=real_c):
    # Function at time t across n x_is
    return np.exp(-sig * ((np.linspace(lo, hi, n) - const * t)**2))


def n_func(xlo, xhi, dx):
    return 1 + int((xhi - xlo) / dx)


def solve(dx, cfl, c=real_c, sigma=real_sigma, pause=False):
    '''
        Returns u at time t=thi
        at np.linspace(xlo, xhi, (xhi - xlo) / dx)

        Returns
        ----
        uk ( u at time t across x )
        err = ||utrue - uk||_inf
    '''
    cs = [23/12, -16/12, 5/12]

    dt = cfl * dx / c
    xs = np.linspace(xlo, xhi, n_func(xlo, xhi, dx))
    uk = np.zeros(len(xs))
    for i in range(1, len(xs)):  # Ignore 0 because BC: uk[0] = 0
        uk[i] = np.exp(-sigma * (xs[i]**2))

    C = sp.diags([-1, 1],
                 [-1, 1],
                 shape=(len(xs), len(xs)),
                 format='csr')
    C[-1, -1] = 2
    C[-1, -2] = -2  # Single sided derivative at other end.
    if pause:
        pdb.set_trace()
    # Boundary condition
    C[0, 1] = 0
    C = (-c / (2 * dx)) * C  # Appropriate scaling.
    fm1 = C @ uk
    fm2 = C @ utrue(xlo, xhi, len(uk), -dt)
    fm3 = C @ utrue(xlo, xhi, len(uk), -2 * dt)
    t = 0.0
    while t < 1.0:
        uk = uk + dt*(cs[0] * fm1 + cs[1] * fm2 + cs[2] * fm3)
        fm3 = fm2
        fm2 = fm1
        fm1 = C @ uk
        t += dt
    error = np.linalg.norm(uk - utrue(xlo, xhi, len(uk), 1.0), ord=np.inf)
    return uk, error


# Part 1:
ntrue = n_func(xlo, xhi, 0.001)
xs = np.linspace(xlo, xhi, ntrue)
u0 = utrue(xlo, xhi, ntrue, 0.0)
u1 = utrue(xlo, xhi, ntrue, 1.0)

dxs = np.array([0.02, 0.01, 0.005, 0.002])
lineargs = ['r-', 'b-', 'c-', 'g-', 'k-']

es = []

for i in range(len(dxs)):
    dx = dxs[i]
    u, e = solve(dx, 0.5)
    es.append(e)
    plt.figure(i)
    plt.plot(np.linspace(xlo, xhi, n_func(xlo, xhi, dx)),
             u,
             'r-o',
             markersize=2.0,
             label='numerical u(x, t=1)')
    plt.plot(xs, u0, 'k--', label='analytical u(x, t=0)')
    plt.plot(xs, u1, 'c-', label='analytical u(x, t=1)')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('u(x, t) vs x for dx=' + str(dx))
    plt.legend(loc='best')
plt.figure(len(dxs))
# pdb.set_trace()
plt.loglog(dxs, es, 'r-', label='error')
plt.loglog(dxs, dxs**1.4, 'k--', label='O(dx**1.4)')
plt.xlabel('dx')
plt.ylabel('error')
plt.title('Error vs dx')
plt.legend(loc='best')
# plt.show()
# pdb.set_trace()

# Part 2
dx = 0.002
cfls = [0.7, 0.75]
for i in range(len(cfls)):
    cfl = cfls[i]
    usol, e = solve(dx, cfl, pause=True)
    # pdb.set_trace()
    plt.figure(1 + len(dxs) + i)
    plt.plot(np.linspace(xlo, xhi, n_func(xlo, xhi, dx)),
             usol,
             'r-o',
             markersize=2.0,
             label='numerical u(x, t=1)')
    plt.plot(xs, u0, 'k--', label='analytical u(x, t=0)')
    plt.plot(xs, u1, 'c-', label='analytical u(x, t=1)')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('u(x, t) vs x for cfl=' + str(cfl))
    plt.legend(loc='best')
plt.show()
