'''
    Uses Kermack-McKendrick model for
    the course of an epidemic in a population.
        y_1^' = -c * y_1 * y_2
        y_2^' = c * y_1 * y_2 - d * y_2
        y_3^' = d * y_2
    y_1 : susceptibles
    y_2 : infectives in circulation
    y_3 : infectives removed by any means

    c : infection rate
    d : removal rate
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

y1 = np.zeros((3,), float)  # The array that the function really wants


def f(y, t, c=1.0, d=5.0):
    '''
        Function to compute the f in
            y' = f(t, y)
        ARGS
        ----
        y : np.array of size (3,) : y at time t
        t : scalar, time step
        c : infection rate
        d : removal rate

        RETURNS
        ----
        f(t, y) as specified by Kermack-McKendrick model
    '''
    return np.array([-c * y[0] * y[1],
                     c * y[0] * y[1] - d * y[1],
                     d * y[1]])


n = 5000  # number of time points in simulation
c = 1.0
d = 1.0
tmin = 0.0
tmax = 1.0
ts_raw = np.polynomial.legendre.leggauss(n)[0]  # This is from -1 to 1
ts = (ts_raw + 1.0) / 2.0
if not(ts[-1] == 1.0):
    ts = np.append(ts, [1.0])
ts = np.linspace(0.0, 1.0, n)
y0 = np.array([95.0, 5.0, 0.0])
ys = odeint(f, y0, ts)
y1 = ys[-1]

plt.plot(ts, ys[:, 0], 'k-', label='y_1 : susceptibles')
plt.plot(ts, ys[:, 1], 'r-', label='y_2 : infectives circulating')
plt.plot(ts, ys[:, 2], 'g--', label='y_3 : infectives removed')
plt.xlabel('t')
plt.ylabel('y_1, y_2, y_3')
plt.legend(loc='best')
plt.title('Populations vs time')
print('y(1.0):', y1)
# import pdb; pdb.set_trace()