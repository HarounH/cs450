import scipy.interpolate as si
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pdb

def f1(x):
    return 1 / (1.0 + 25 * x**2)

def f2(x):
    return np.exp(np.cos(x))


def cheby_points(n):
    return np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))


# n=12 for f1
sample_density_multiplier = 10
n = 12
marker_size = 6
u_node = np.linspace(-1, 1, n)
u_samp = np.linspace(-1, 1, n * sample_density_multiplier)
u_n_f1 = f1(u_node)
u_s_f1 = f1(u_samp)
c_node = cheby_points(n)
c_samp = cheby_points(n * sample_density_multiplier)
c_n_f1 = f1(c_node)
# c_s_f1 = f1(c_samp)

lu1 = si.lagrange(u_node, u_n_f1)
cu1 = si.CubicSpline(u_node, u_n_f1)
lc1 = si.lagrange(c_node, c_n_f1)
plt.figure(1)
plt.plot(u_samp, u_s_f1, 'k-', label='f1')
plt.scatter(u_node, lu1(u_node), marker='o', c='r', label='uniform lagrange nodes')
plt.scatter(u_node, cu1(u_node), marker='s', c='g', label='cubic spline nodes')
plt.scatter(c_node, lc1(c_node), marker='*', c='b', label='chebyshev lagrange nodes')
plt.plot(u_samp, lu1(u_samp), 'r--', markersize=marker_size, label='uniform lagrange')
plt.plot(u_samp, cu1(u_samp), 'g--', markersize=marker_size, label='cubic spline')
plt.plot(c_samp, lc1(c_samp), 'b--', markersize=marker_size, label='chebyshev lagrange')
plt.xlabel('x')
plt.ylabel('function value')
plt.title('f(x) vs x: interpolants and original')
plt.legend(bbox_to_anchor=(1.05, 1.0))

plt.show()
exit()
print('From figure 1 we see that using lagrange interpolation on uniformly spaced points produces large errors, especially near the boundary. The other two interpolants are significantly better near the boundary. We also see that the cubic spline is better able to approximate the function. This seems to suggest that the cubic spline generalizes (a concept from machine learning) better than the lagrange polynomial. ')

# Next part
# f1
n = 50
sample_density_multiplier = 10
u_samp = np.linspace(-1, 1, n * sample_density_multiplier)
c_samp = cheby_points(n * sample_density_multiplier)
samp = np.unique(np.concatenate((u_samp, c_samp)))
samp.sort()
y_s = f1(samp)
# y_u_samp = f1(u_samp)
# y_c_samp = f1(c_samp)
plt.figure(2)

ns = list(range(4, 51))
elu1 = []
ecu1 = []
elc1 = []
for ni in ns:
    u_node = np.linspace(-1, 1, ni)
    c_node = cheby_points(ni)
    y_un = f1(u_node)
    y_cn = f1(c_node)
    lu1 = si.lagrange(u_node, y_un)
    cu1 = si.CubicSpline(u_node, y_un)
    lc1 = si.lagrange(c_node, y_cn)

    # Now to measure error
    elu1.append(la.norm(y_s - lu1(samp)))
    ecu1.append(la.norm(y_s - cu1(samp)))
    elc1.append(la.norm(y_s - lc1(samp)))


plt.semilogy(ns, elu1, 'r--o', label='uniform lagrange')
plt.semilogy(ns, ecu1, 'g--s', label='cubic spline')
plt.semilogy(ns, elc1, 'b--*', label='chebyshev lagrange')
plt.xlabel('n')
plt.ylabel('log(error)')
plt.title('f1 : log(error) vs n for interpolants')
plt.legend(loc='best')
# plt.show()

print('We plot log(||error||) vs n in the next two plots.')

print('For function f1, the cubic spline is the best interpolant, followed by the lagrange interpolant using chebyshev nodes. The lagrange interpolant using uniformly spaced performs the worst. Notice that at low n, the error is relatively small for all 3. However, the difference between the interpolants becomes clear at larger n. At larger n, the instability of lagrange causes even the chebyshev variant to have large error.')

# f2
n = 50
sample_density_multiplier = 10
u_samp = np.linspace(0, 2 * np.pi, n * sample_density_multiplier)
c_samp = np.pi * (1 + cheby_points(n * sample_density_multiplier))
# y_u_samp = f2(u_samp)
# y_c_samp = f2(c_samp)
samp = np.unique(np.concatenate((u_samp, c_samp)))
samp.sort()
y_s = f2(samp)
elu2 = []
ecu2 = []
elc2 = []
for ni in ns:
    u_node = np.linspace(0, 2 * np.pi, ni)
    c_node = np.pi * (1 + cheby_points(ni))
    y_un = f2(u_node)
    y_cn = f2(c_node)
    lu2 = si.lagrange(u_node, y_un)
    cu2 = si.CubicSpline(u_node, y_un)
    lc2 = si.lagrange(c_node, y_cn)

    elu2.append(la.norm(y_s - lu2(samp)))
    ecu2.append(la.norm(y_s - cu2(samp)))
    elc2.append(la.norm(y_s - lc2(samp)))
plt.figure(3)
plt.semilogy(ns, elu2, 'r--o', label='uniform lagrange')
plt.semilogy(ns, ecu2, 'g--s', label='cubic spline')
plt.semilogy(ns, elc2, 'b--*', label='chebyshev lagrange')
plt.xlabel('n')
plt.ylabel('log(error)')
plt.title('f2 : log(error) vs n for interpolants')
plt.legend(loc='best')
plt.show()

print('For function f2, lagrange interpolation with chebyshev nodes always performs better than with uniformly spaced points. At high n, lagrange is unstable and hence the cubic spling is a better fit. At mid range n (10-20), lagrange interpolation with chebyshev points performs better than cubic splines. At low n ( < 10), lagrange interpolation with chebyshev nodes still performs better.')
