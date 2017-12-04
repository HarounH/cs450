import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (1.0 + (1.0 / x))**x


ns = range(100, 1000)
anss = np.abs(np.array(list(map(f, ns))) - np.e)

fac1 = anss[0] / (ns[0]**-0.5)
fac2 = anss[0] / (ns[0]**-1)
fac3 = anss[0] / (ns[0]**-2)
fac4 = anss[0] / (ns[0]**-3)

plt.figure(1)
plt.plot(ns, fac1 * np.array(list(map(lambda x: x**-0.5, ns))), 'b-', label='-0.5')
plt.plot(ns, fac2 * np.array(list(map(lambda x: x**-1, ns))), 'r-', label='-1')
plt.plot(ns, fac3 * np.array(list(map(lambda x: x**-2, ns))), 'y-', label='-2')
plt.plot(ns, fac4 * np.array(list(map(lambda x: x**-3, ns))), 'c-', label='-3')
plt.plot(ns, anss, 'k--', label='real')

plt.xlabel('n')
plt.ylabel('n^-k')
plt.title('Sanity check')
plt.legend(loc='best')

plt.show()
