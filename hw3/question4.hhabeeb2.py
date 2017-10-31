#!/usr/bin/python
'''
    This file implements inverse iteration with shifts
'''

__author__ = 'Haroun Habeeb'
__mail__ = 'hhabeeb2@illinois.edu'

import numpy as np
import matplotlib.pyplot as plt
import pdb


def matrix(n):
    return np.array([[
        (1.0 if (i == j) else (-1.0 if (j > i) else 0))
        for j in range(0, n)]
        for i in range(0, n)],
        dtype=np.float)


vals_svd = []
vals_eig = []
szs = list(range(5, 100, 5))
all_singular_values = []
all_eigs = []
for sz in szs:
    A = matrix(sz)
    _, singular_values, _ = np.linalg.svd(A)
    eigs = np.linalg.eigvals(np.transpose(A) @ A)
    # pdb.set_trace()
    vals_svd.append((singular_values[0]) / (singular_values[-1]))
    vals_eig.append(np.sqrt(eigs[0] / eigs[-1]))
    all_eigs.append(eigs)
    all_singular_values.append(singular_values)
    print('sz=', sz, end='\r')
# pdb.set_trace()
# fig, axarr = plt.subplots(2, sharex=True)
# axarr[0].plot(szs, vals_svd, 'r', label='using svd')
# axarr[0].plot(szs, vals_eig, 'b', label='using eigvals')
# axarr[0].set_xlabel('matrix size')
# axarr[0].set_ylabel('\sigma_{max} / \sigma_{min}')
# axarr[0].set_title('\sigma_{max} / \sigma_{min} vs matrix size SVD vs eigvals')
# axarr[0].legend(loc='best')

# axarr[1].plot(szs, vals_eig, 'b', label='using eigvals')
# axarr[1].set_xlabel('matrix size')
# axarr[1].set_ylabel('\sigma_{max} / \sigma_{min}')
# axarr[1].set_title('\sigma_{max} / \sigma_{min} vs matrix size for eigvals')
# axarr[1].legend(loc='best')

plt.plot(szs, vals_eig, 'b', label='using eigvals')
plt.xlabel('matrix size')
plt.ylabel('\sigma_{max} / \sigma_{min}')
plt.title('\sigma_{max} / \sigma_{min} vs matrix size for eigvals')
plt.legend(loc='best')

# fig.subplots_adjust(vspace=1.0)
plt.tight_layout()
plt.show()
pdb.set_trace()
