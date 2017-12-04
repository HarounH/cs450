import numpy as np
import pdb

def trap0(x, f):
    ans = 0.0
    for i in range(0, len(x)):
        ans += f[i] * (0.5 if ((i == 0) or i == (len(x) - 1)) else 1.0)
    ans *= x[1] - x[0]
    return ans


def trap1(x, f):
    ans = 0.0
    for i in range(0, len(x)):
        if ((i % 2) == 0):
            ans += f[i] * (0.5 if ((i == 0) or i == (len(x) - 1)) else 1.0)
    ans *= x[1] - x[0]
    return ans


def trap2(x, f):
    return (f[0] + f[-1]) / 2

x = [0, 1.0 / 4, 0.5, 0.75, 1.0]
f1 = [0, 0.01944224, -0.10295637, -0.28199532, -0.45610778]
f2 = [xi**3 for xi in x]
f3 = [xi**5 for xi in x]

Ih1 = (trap0(x, f1))
Ih2 = (trap0(x, f2))
Ih3 = (trap0(x, f3))
I4h1 = (trap1(x, f1))
I4h2 = (trap1(x, f2))
I4h3 = (trap1(x, f3))
I8h1 = (trap2(x, f1))
I8h2 = (trap2(x, f2))
I8h3 = (trap2(x, f3))

Ir1 = ((4 * trap0(x, f1) - trap1(x, f1)) / 3)
Ir2 = ((4 * trap0(x, f2) - trap1(x, f2)) / 3)
Ir3 = ((4 * trap0(x, f3) - trap1(x, f3)) / 3)

T = {}
T[0] = {0: I8h1}
T[1] = {0: I4h1}
T[2] = {0: Ih1}

T[1][1] = (4 * T[1][0] - T[0][0])/3
T[2][1] = (4 * T[2][0] - T[1][0])/3
T[2][2] = (16 * T[2][1] - T[1][1])/15

print(T[2][2])



T = {}
T[0] = {0: I8h2}
T[1] = {0: I4h2}
T[2] = {0: Ih2}

T[1][1] = (4 * T[1][0] - T[0][0])/3
T[2][1] = (4 * T[2][0] - T[1][0])/3
T[2][2] = (16 * T[2][1] - T[1][1])/15

print(T[2][2])



T = {}
T[0] = {0: I8h3}
T[1] = {0: I4h3}
T[2] = {0: Ih3}

T[1][1] = (4 * T[1][0] - T[0][0])/3
T[2][1] = (4 * T[2][0] - T[1][0])/3
T[2][2] = (16 * T[2][1] - T[1][1])/15

print(T[2][2])

pdb.set_trace()