#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:56:18 2018

@author: mikal
"""
from sympy import *
import numpy.random
import matplotlib.pyplot as plt
import numpy.linalg

N = 2
D = int(N*(N+1)/2) + N

X = symbols("x1 x2 x3 x4 x5")
x1, x2, x3, x4, x5 = x[:]

A = np.array([[x1, x2], [x2, x3]])
c = np.array([x4, x5])
z = np.array([0.5, 0.5])

r = (z-c).T*A*(z-c)
gradr = [diff(r, x) for x in X]
for g in gradr:
    g.doit()
print(gradr)


def generate_points(num, dim=2):
    z = np.random.rand(num, dim)
    return z
