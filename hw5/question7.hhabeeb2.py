'''
    HW5 Q7 of CS450 Fall 2017
    Analyses stability of Euler Forward
        y' = Ay

    A = [[0, 0, -5],
         [1, 0, -9.25],
         [0, 1, -6]]

    y0 = (3, 3, 3)

'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import matplotlib.pyplot as plt
import pdb

A = np.array([[0, 0, -5],
              [1, 0, -9.25],
              [0, 1, -6]])

y0 = np.array([3, 3, 3], dtype=float)

lambdas = np.linalg.eig(A)[0]
dt_max = 0.5
print('Working given in comments - max dt=', dt_max)
# print('Eigs=', lambdas, ' are complex.')
'''
    Eigs:
        -1 + 0.5j
        -1 - 0.5j
        -4
    for lambda = -1 + 0.5j
    |1 + h\lambda| = |1-h + 0.5hj| = 1 + 1.25h^2 -2h < 1
        \Rightarrow h(1.25h - 2) < 0
        \Rightarrow h \in [0, 1.6]

    for lambda = -1 - 0.5j,
    similar analysis applies since the complex part's sign doesn't matter

    for lambda = -4, we have h \in [0, 0.5]
'''
eps = 0.01
dt_a = dt_max + eps
dt_b = dt_max - eps

tmin = 0.0
tmax = 20.0
t_a = 0.0
t_b = 0.0
t_as = [0.0]
t_bs = [0.0]

y_as = [y0]
y_bs = [y0]

while (t_a <= (tmax)) or (t_b <= tmax):
    t_a += dt_a
    t_b += dt_b
    if t_a <= tmax:
        t_as.append(t_a)
        y_as.append(y_as[-1] + dt_a * A @ y_as[-1])
    if t_b <= tmax:
        t_bs.append(t_b)
        y_bs.append(y_bs[-1] + dt_b * A @ y_bs[-1])
    if (t_a >= tmax) and (t_b >= tmax):
        break
y_as = np.array(y_as)
y_bs = np.array(y_bs)

# Single plot
plt.plot(t_as, y_as[:, 0], 'r-', label='y_0 : dt_max + eps')
plt.plot(t_as, y_as[:, 1], 'b-', label='y_1 : dt_max + eps')
plt.plot(t_as, y_as[:, 2], 'y-', label='y_2 : dt_max + eps')

plt.plot(t_bs, y_bs[:, 0], 'r--', label='y_0 : dt_max - eps')
plt.plot(t_bs, y_bs[:, 1], 'b--', label='y_1 : dt_max - eps')
plt.plot(t_bs, y_bs[:, 2], 'y--', label='y_2 : dt_max - eps')

plt.ylabel('y')
plt.xlabel('t')
plt.title('Comparing stability of euler forward')
plt.legend(loc='best')
plt.show()
