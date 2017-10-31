import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import pdb
'''
    Finding out which points to sample from f(t)
    so that we can best interpoate it using a
    polynomial.

    This is done using LM, solving a non linear
    least squares problem

    Game:
        min_{x1, x2... x_n} \bar{q}^T\bar{q}

        where:
            \bar{q} = [q(t_1), q(t_2)... q(t_i)... q(t_m)]
            q = \Pi_{j=1}^{n} (t - x_j)
'''


def get_t(m):
    h = 2.0 / m
    return np.array([-1 - h / 2 + i * h for i in range(1, m + 1)])


def q(xk, t):
    '''
        Computes the residual function.
    '''
    return np.prod(t[:, None] - xk, axis=1)
    # pdb.set_trace()
    # def func_eval(ti):
        # return reduce(lambda x, y: x * y, (ti - xk), 1)
    # vfunc_eval = np.vectorize(func_eval)
    # return np.array([func_eval(ti) for ti in t])


def J(xk, t):
    '''
        Computes jacobian
    '''
    jacobian = np.zeros((len(t), len(xk)))
    mask = np.ones(xk.shape, dtype=bool)

    for i in range(0, len(xk)):
        mask[i] = False
        jacobian[:, i] = -q(xk[mask], t)
        mask[i] = True
    # muls = -q(xk, t)
    # for row in range(0, len(t)):
    #     for col in range(0, len(xk)):
    #         if t[row] - xk[col] != 0:
    #             jacobian[row, col] = muls[row] / (t[row] - xk[col])
    return jacobian


def lm_step(xk, Jxk, uk, qk):
    A = np.concatenate((Jxk, np.sqrt(uk) * np.eye(len(xk))))
    b = np.concatenate((-qk, np.zeros(len(xk))))
    return xk + np.linalg.lstsq(A, b)[0]


def lm(m, n, t, ufunc=lambda k: np.exp(-k), eps=10**-15):
    # print('Starting n=', n)
    # t = get_t(m)
    # xks = np.array([i / (n - 1) for i in range(0, n)])  # Uniformly space
    xks = [np.linspace(-1.0, 1.0, n)]
    not_converged = True

    while not_converged:
        # Do stuff
        qk = q(xks[-1], t)
        Jxk = J(xks[-1], t)
        uk = ufunc(len(xks) - 1)
        xks.append(lm_step(xks[-1], Jxk, uk, qk))
        # check for convergence
        if (np.linalg.norm(xks[-2] - xks[-1]) <= eps):
            not_converged = False
            break
        if len(xks) >= 50:
            not_converged = True
            break
    return xks


m = 300
# For n = 40
ts = get_t(m)
xks40 = lm(m, 40, ts)
q0 = q(xks40[0], ts)  # [[0] for t in ts]
qf = q(xks40[-1], ts)  #[[0] for t in ts]

# q0 vs qf
plt.figure(1)
plt.plot(ts, q0, 'r-', label='uniformly')
plt.plot(ts, qf, 'b-', label='optimized')
plt.title('q vs t: uniformly spaced nodes vs optimized nodes')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('q(t)')
plt.gca().set_xlim([-1.0, 1.0])
# plt.show()


# by varying n
plt.figure(2)
q0 = [np.linalg.norm(q(np.linspace(-1, 1, n), ts)) for n in range(1, 41)]
qf = [np.linalg.norm(q(lm(m, n, ts)[-1], ts)) for n in range(1, 41)]
plt.semilogy(range(1, 41), q0, 'r-', label='uniformly')
plt.semilogy(range(1, 41), qf, 'b-', label='optimized')
plt.title('log(||q||) vs n: uniformly vs optimized')
plt.xlabel('n')
plt.ylabel('log(||q||)')
plt.legend(loc='best')
plt.gca().set_xlim([1, 40])
# plt.show()

print('At n=40, log(error_{optimized}) is 10**-10, while the uniformly spaced points have an log(error) of 10**-5. It is not clear what the theoretical improvement is. However, the relation of residual with respect to number of nodes appears to go from e^(-nz) for uniformly spaced nodes to e^(-n) for optimally spaced points where z is a number between 0, 1')

# gauss vs lm
plt.figure(3)
qgm = [np.linalg.norm(q(lm(m, n, ts, ufunc=lambda k: 0)[-1], ts)) for n in range(1, 41)]
plt.semilogy(range(1, 41), q0, 'r-', label='uniformly')
plt.semilogy(range(1, 41), qf, 'b-', label='optimized')
plt.semilogy(range(1, 41), qgm, 'c-', label='gauss-newton')
plt.title('log(||q||) vs n: uniformly vs optimized vs gauss-newton')
plt.xlabel('n')
plt.ylabel('log(||q||)')
plt.legend(loc='best')
plt.gca().set_xlim([1, 40])
# plt.show()

plt.show()

print('The LM method converges even as dimensionality increases. Gauss-Newton on the other hand, fails for higher dimensions. The last plot shows that the maximum error when using nodes optimized using gauss newton aren\'t good.')
