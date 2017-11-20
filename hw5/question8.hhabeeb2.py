'''
    Stability of unsteady diffusion


        \frac{du}{dt} = Lu + f
        u(t=0) = u_0

    Solve in t \in [0, 10]
    Use euler forward with dt = 0.01

    Note: u is a vector. It has length n


    To answer what is the largest n, we must say:
        argmax_n {n | min_k (dx * dx) / (v * (1 - cos(pi * k* dx))) > dt}
        = argmax_n {n | dx * dx / v * (1 - cos(pi * n / (n + 1)) > dt}
    (min_k occurs at largest k, i.e., k=n)
    (1 / (n + 1)^2) / (v * (1 - cos(pi * n / (n + 1))) > 0.01


    minima occurs when
        tan (pi * n / (2 * (n + 1))) = -(n + 1) / 4
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import pdb
import numpy as np
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def simulate(u0, f, v=0.05, dt=0.01, n=20, tmin=0.0, tmax=10.0):
    '''
        Performs euler forward to compute u
        Using diffusion equation
            u' = Lu + f

        ARGS
        -----
            u0 : (n,) array initial condition
            f : function (t, y) -> (n,) array
            v : scalar, diffusion rate
            dt: scalar, time step
            n: scalar, number of grid points
            tmin: scalar, starting time
            tmax: scalar, ending time
        RETURNS
        -----
            us: list of values of u at time steps
            ts: list of time steps
            xs: list of grid points
    '''
    # pdb.set_trace()
    dx = 1.0 / (n + 1)
    L = -v * ((n + 1)**2) * sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    xs = np.array(list(range(n))) * dx
    k = np.array(range(1, n + 1))
    lambda_k = - 2 * v * (1 - np.cos(np.pi * k * dx)) / (dx**2)
    dt_max = (-2.0 / lambda_k).min()
    # print('Maximum allowed dt=', dt_max)
    us = [u0]
    ts = [tmin]
    t = tmin
    while t < tmax:  # simulation until t is complete.
        us.append(us[-1] + dt * ((L @ us[-1]) + f(t, us[-1])))
        t += dt
        ts.append(t)
    return us, ts, xs


def f(t, y):
    return np.ones((len(y),))


n = 20
v = 0.05
tmin = 0.0
tmax = 10.0

us, ts, xs = simulate(np.zeros((n,)), f, tmin=tmin, tmax=tmax, n=n)
us = np.array(us)
un20t10 = us[-1]
plt.figure(1)
plt.plot(xs, us[0], 'r-', label='At t=0')
plt.plot(xs, us[len(us) // 2], 'c-', label='At t=' + str(ts[len(us) // 2]))
plt.plot(xs, us[-1], 'y-', label='At t=10.0')
plt.xlabel('x')
plt.ylabel('u')
plt.legend(loc='best')
plt.title('u vs x for n=' + str(n))
ylims = plt.ylim()


n = 40
v = 0.05
tmin = 0.0
tmax = 1.0

us, ts, xs = simulate(np.zeros((n,)), f, tmin=tmin, tmax=tmax, n=n)
us = np.array(us)
un40t10 = us[-1]


# for n=30, n=31
n = 30
v = 0.05
tmin = 0.0
tmax = 10.0

us, ts, xs = simulate(np.zeros((n,)), f, tmin=tmin, tmax=tmax, n=n)
us = np.array(us)
plt.figure(2)
plt.plot(xs, us[0], 'r-', label='At t=0 n=30')
plt.plot(xs, us[len(us) // 2], 'c-', label='At t=' + str(ts[len(us) // 2]) + ' n=30')
plt.plot(xs, us[-1], 'y-', label='At t=10.0 n=30')
# plt.xlabel('x')
# plt.ylabel('u')
# plt.ylim(ylims)
# plt.legend(loc='best')
# plt.title('u vs x for n=30,31')

# for n = 31
n = 31
v = 0.05
tmin = 0.0
tmax = 10.0

us, ts, xs = simulate(np.zeros((n,)), f, tmin=tmin, tmax=tmax, n=n)
us = np.array(us)

plt.plot(xs, us[0], 'r--', label='At t=0 n=31')
plt.plot(xs, us[len(us) // 2], 'c--', label='At t=' + str(ts[len(us) // 2]) + ' n=31')
plt.plot(xs, us[-1], 'y--', label='At t=10.0 n=31')
plt.xlabel('x')
plt.ylabel('u')
plt.ylim(ylims)
plt.legend(loc='best')
plt.title('u vs x for n=30,31')


print('n=20 : ||u(t=10)||_{\\infty}=', np.linalg.norm(un20t10, ord=np.inf))
print('n=40 : ||u(t=10)||_{\\infty}=', np.linalg.norm(un40t10, ord=np.inf))
print('When we set n=40, the infinity norm of $$u$$ increases from 2.4 to 1.07e32. It is no longer stable.')
print('n=30 is the largest permissible value. when v=0.05 and dt=0.01')
# plt.show()