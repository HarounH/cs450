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

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt



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
    '''
    L = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    dx = 1.0 / (n + 1)
    k = np.array(range(1, n + 1))
    lambda_k = - 2 * v * (1 - np.cos(np.pi * k * dx)) / (dx**2)
    dt_max = (-2.0 / lambda_k).min()
    print('Maximum allowed dt=', dt_max)
    us = [u0]
    ts = [tmin]
    t = tmin
    while t <= tmax:
        us.append(us[-1] + dt * L @ us[-1] + f(t, us[-1]))
        t += dt
    return us, ts


f = lambda t, y: np.zeros((n,))
n = 20
v = 0.05
tmin = 0.0
tmax = 10.0

us, ts = simulate(np.zeros((n,)), f)

