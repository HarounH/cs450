#!/usr/bin/python
'''CS450 Fall 2017 Quiz 6 things'''
__author__ = 'harun habeeb'
__mail__ = 'hhabeeb2@illinois.edu'
# import sys
import numpy as np
import pdb

A = np.array([[-1, 1, 1], [1, 1, 0], [-1, 1, 1], [1, 1, 0]], dtype=np.float)
b = np.array([[1], [2], [3], [4]], dtype=np.float)
pdb.set_trace()
x = np.dot(np.linalg.pinv(A), b)
y = np.dot(np.dot(A, np.linalg.pinv(A)), b)
print(x)
print(y)
