__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pdb

n = 10

A = 5 * np.random.rand(n, n)
L = {}
L[1] = -A.T @ A
L[2] = -(A.T + A)
L[3] = -(A.T - A)

u0 = np.random.rand(n)
ts = np.linspace(0, 0.1, 100)
sol = {}
f = {}
for i in L:
    f[i] = lambda u, t: L[i] @ u
    sol[i] = odeint(f[i], u0, ts)
pdb.set_trace()
plt.figure(1)
plt.plot(ts, [np.linalg.norm(x) for x in sol[1]], 'r-', label='1')
plt.plot(ts, [np.linalg.norm(x) for x in sol[2]], 'b-', label='2')
plt.plot(ts, [np.linalg.norm(x) for x in sol[3]], 'k--', label='3')
plt.legend(loc='best')
plt.show()
