'''
'''



# Problem set up code
import numpy as np


def ydata(x):
    t = 20 * np.random.rand(200, 1)  #Generate random points on interval
    t = np.sort(t, axis=0)  # Sort the points
    #Evaluate function at points
    y = x[0, 0] * np.exp(-x[1, 0] * t) * np.sin(x[2, 0] * t - x[3, 0]) + x[4, 0]
    return y, t

a = 0.3 + 2 * (np.random.rand() - 0.5) / 2
b = 0.1 + 2 * (np.random.rand() - 0.5) / 25
omega = 4 + 2 * (np.random.rand() - 0.5) / 2
phase = -1.0 + (2 * (np.random.rand() - 0.5)) / 2
c = 1.0 + (2 * (np.random.rand() - 0.5)) / 2
coeffs = np.array([a, b, omega, phase, c])
#coeffs = np.array([0.3, 0.1, 4.0, -1.0, 1.0])

coeffs = coeffs.reshape((5, 1))
[y, t] = ydata(coeffs)


# Solution begins here
import matplotlib.pyplot as plt
import numpy as np

print(y)  #Your code can access these provided  n x 1 numpy arrays
print(t)

#You can use this as your initial guess.
x0 = np.array([0.3, 0.1, 4.0, -1.0, 1.0])
# x0 = x.reshape((5, 1))


def f(t, xin):
    '''
        t: np array of points
    '''
    return xin[4] + xin[0] * np.exp(-xin[1] * t) * np.sin(xin[2] * t - xin[3])


'''
    Jacobian:
        J[:, 0] = np.exp(-x[1] * t) * np.sin(x[3] + x[2] * t)
        J[:, 1] = x[0] * (-t) * np.exp(-x[1] * t) * np.sin(x[3] + x[2] * t)
        J[:, 2] = x[0] * np.exp(-x[1] * t) * t * np.cos(x[2] * t + x[3])
        J[:, 3] = x[0] * np.exp(-x[1] * t) * np.cos(x[2] * t + x[3])
        J[:, 4] = 1
'''


def lm(t, y, x0, eps=10**-6):
    xs = [x0]
    not_converged = True
    m = len(t)
    I = np.eye(len(x0), dtype=np.float_)
    zs = np.zeros((len(x0), 1))
    while True:
        print('Iteration', len(xs))
        xk = xs[-1]
        sin = np.sin(xk[2] * t - xk[3])
        cos = np.cos(xk[2] * t - xk[3])
        exp = np.exp(-xk[1] * t)
        f = xk[4] + xk[0] * exp * sin
        r = y - f  # residual
        # pdb.set_trace()
        j = -np.concatenate(((exp * sin),
                            (-xk[0] * t * exp * sin),
                            (xk[0] * exp * t * cos),
                            (-xk[0] * exp * cos),
                            np.ones((m, 1))),
                           axis=1)
        uksqrt = np.linalg.norm(r)
        A = np.concatenate((j, uksqrt * I))
        b = np.concatenate((-r, zs))
        sk = np.linalg.lstsq(A, b)[0]
        xs.append(xk + sk[:, 0])
        if uksqrt < eps:
            break
        print('norm_r', uksqrt)
    return xs[-1], np.linalg.norm(r)


x, norm_r = lm(t, y, x0)
x = x.reshape((5, 1))


n_samples = 100
tlinspace = np.linspace(min(t), max(t), n_samples)
tlinspace = np.concatenate((tlinspace[:, None], t), axis=0)
tlinspace = tlinspace[:, 0]
tlinspace.sort()
tlinspace = tlinspace[:, None]
predy = f(tlinspace, x)
predy_scatter = f(t, x)

plt.figure(1)
plt.plot(tlinspace, predy, 'k--', label='model')
plt.scatter(t, y, s=100, c='r', marker='o', label='observed nodes')
plt.scatter(t, predy_scatter, s=64, c='b', marker='h', label='model nodes')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('f')
plt.title('y vs t')
plt.show()
