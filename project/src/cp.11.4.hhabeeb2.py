'''
    Use method of lines and any ODE solve of your choice to solve
        \(u_t = -u_x\) with \(x \in [0, 1], t \gteq 0\)
    initial conditions:
        u(0, x) = 0
    boundary conditions:
        u(t, 0) = 1

    Over \(t \in [0, 2]\)
    Plot u(t,x) vs t , x (3D surface plot)

    for spatial discretization, try both:
        one sided
        centered finite difference

    The actual solution is a step function of height 1 moving toward +x
    with velocity 1

    Answer:
        Does either scheme get close?
        Describe the difference between computed solutions
        Which soln is smoother?
        Which is more accurate?
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'


import sys
import numpy as np
from scipy.integrate import odeint
import scipy.sparse as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter


# Parameters
eps = 10**-6

# number of points along x axis
nx = 1000 if len(sys.argv) < 2 else int(sys.argv[1])
nt = 1000 if len(sys.argv) < 3 else int(sys.argv[2])

# dx = 1.0 / (nx - 1)
xs = np.linspace(0, 1.0, nx)

dx = xs[1] - xs[0]
obdx = 1 / dx
tmin = 0.0
tmax = 2.0
dt = (tmax - tmin) / nt
ts = np.linspace(tmin, tmax, nt)

mesh_xy, mesh_yx = np.meshgrid(xs, ts)
# y0 = [u(0, x_i) for x_i in xs]
y0 = np.zeros((nx,))
y0[0] = 1.0

# Now, our system of equations is a \(nx\) dimensional one.
# y' = [u_t(t, x_i) for x_i in xs]
# u_x(t, x_i) = ...
# one sided: u_x(t, x_i) = (u(t, x_{i+1}) - u(t, x_i)) / (x_{i+1} - x_i)
#       0 <= i < nx-1
#       u_x(t, x_{nx}) = ???
# Centered: u_x(t, x_i) = (u(t, x_{i+1}) - u(t, x_{i-1})) / (x_{i+1} - x_{i-1})
#       0 < i < nx-1
#       u_x(t, 0) = ???
#       u_x(t, 1) = ???
#
# the equivalent ODE is
# u_t(t, x_i) = np.array(coefficients_from_rule, shape=(1, nx)) * u(t, x)
# equivalently,
# y' = Jy
# The two methods above differ on their definition of J

fig = plt.figure(1)
ax = fig.gca(projection='3d')

# solve using one sided
coeffs_os = sp.diags([-obdx * np.ones((nx,)),
                      obdx * np.ones((nx - 1,))],
                     [0, -1],
                     dtype=float)


def get_os_yp(y, t):
    ans = (coeffs_os * y)
    ans[0] = 0.0
    return ans


os_sol = odeint(get_os_yp,
                y0,
                ts)
surf = ax.plot_surface(mesh_xy, mesh_yx, os_sol, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(t, x)')
ax.set_title('Using one sided')

# solve using centered
upper_diag = -0.5 * obdx * np.ones((nx - 1,))
upper_diag[0] = -obdx
lower_diag = 0.5 * obdx * np.ones((nx - 1,))
lower_diag[-1] = obdx
diag = np.zeros((nx,))
diag[0] = obdx
diag[-1] = -obdx
coeffs_cd = sp.diags([lower_diag, diag, upper_diag],
                     [-1, 0, 1],
                     dtype=float)


def get_cd_yp(y, t):
    ans = coeffs_cd * y
    ans[0] = 0
    return ans


fig = plt.figure(2)
ax = fig.gca(projection='3d')


cd_sol = odeint(get_cd_yp,
                y0,
                ts)
# pdb.set_trace()
surf = ax.plot_surface(mesh_xy, mesh_yx, cd_sol, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(t, x)')
ax.set_title('Using centered difference')


plt.show()
