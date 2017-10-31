import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import pdb
import matplotlib.pyplot as plt

######## EXAMPLE FOR USING MINIMIZE_SCALAR ##############
# Define function
def f(x,y,s):
    return s*(y+x)
# Call routine - min now contains the minimum x for the function
# min = opt.minimize_scalar(f,args=(y,s)).x

#########################################################


def rosenbrock(x):
    x1 = x[0]
    x2 = x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2


# FINISH THIS
def gradient(x):
    # Returns gradient of rosenbrock function at x as numpy array
    x1 = x[0]
    x2 = x[1]
    grad = np.array([-400.0 * (x2 - x1**2) * x1 - 2.0 * (1.0 - x1),
                     200.0 * (x2 - x1**2)])
    return grad


# FINISH THIS
def hessian(x):
    # Returns hessian of rosenbrock function at x as numpy array
    x1 = x[0]
    x2 = x[1]
    hess = np.array([[2.0 - (400.0 * x2) + (1200.0 * (x1**2)), (-400.0 * x1)],
                     [(-400.0 * x1), (200.0)]])
    return hess


# INSERT NEWTON FUNCTION DEFINITION
def newton_step(xk, grad, hess):
    return xk - np.linalg.solve(hess, grad)


# INSERT STEEPEST DESCENT FUNCTION DEFINITION
def steepest_descent_step(xk, grad, hess):
    alphak = opt.minimize_scalar(
        lambda alpha: rosenbrock(xk - alpha * grad)).x
    return xk - alphak * grad


def optimize(step_func, start, grad=gradient, hess=hessian, niter=10):
    xs = [start]
    while len(xs) < niter + 1:
        xs.append(step_func(xs[-1], grad(xs[-1]), hess(xs[-1])))
    return xs[-1], xs


# DEFINE STARTING POINTS AND RETURN SOLUTIONS
start1 = np.array([-1., 1.])
start2 = np.array([0., 1.])
start3 = np.array([2., 1.])
nm1, nms1 = optimize(newton_step, start1)
nm2, nms2 = optimize(newton_step, start2)
nm3, nms3 = optimize(newton_step, start3)
sd1, sds1 = optimize(steepest_descent_step, start1)
sd2, sds2 = optimize(steepest_descent_step, start2)
sd3, sds3 = optimize(steepest_descent_step, start3)

plt.plot([x[0] for x in nms1], [x[1] for x in nms1], 'r-*', label='nms1')
plt.plot([x[0] for x in nms2], [x[1] for x in nms2], 'g-*', label='nms2')
plt.plot([x[0] for x in nms3], [x[1] for x in nms3], 'b-*', label='nms3')
plt.plot([x[0] for x in sds1], [x[1] for x in sds1], 'r--o', label='sds1')
plt.plot([x[0] for x in sds2], [x[1] for x in sds2], 'g--o', label='sds2')
plt.plot([x[0] for x in sds3], [x[1] for x in sds3], 'b--o', label='sds3')
plt.scatter([1], [1], c='k', marker='s')
plt.legend(loc='best')
plt.show()
