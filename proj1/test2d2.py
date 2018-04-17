#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:21:45 2018

@author: mikal
"""

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import numpy.linalg

plt.axis('equal')
plt.style.use('ggplot')

N = 2

def generate_points(num, dim=2):
    z = 2*np.random.rand(num, dim) - (1, 1)
    w = np.random.choice([-1, 1], 10)
    return z, w


def xTm(x, n):
    A = np.zeros((n, n))
    b = int(n*(n+1)/2)
    for i in range(n):
        for j in range(i, n):
            ind = b - int((n-i)*(n-i+1)/2) + j-i
            A[i, j] = x[ind]
            A[j, i] = x[ind]
    c = x[b:]
    return A, c


def plot_ellipse(x, n):
    A, c = xTm(x, n)
    w, v = np.linalg.eig(A)
    t = np.linspace(0, 2*np.pi, 100)
    z_star = np.reshape([np.cos(t), np.sin(t)], (2, 100)).T
    z = z_star@v/np.sqrt(w) + c
    plt.plot(z[:, 0], z[:, 1])


def residuals(z, w, x, n):
    m = len(z)
    res = np.zeros((m, ))
    A, c = xTm(x, n)
    for i in range(m):
        res[i] = ((z[i] - c).T@A@(z[i] - c)-1)*w[i]
        if (res[i] < 0):
            res[i] = 0
    return res


def f(z, w, x, n, printres=False):
    res = residuals(z, w, x, n)
    if printres: print(w); print(res)
    return np.sum(res**2)


def gradr(zi, w, x, n):
    A, c = xTm(x, n)
    gr = np.ones((len(x), ))
    lenA = int(n*(n+1)/2)
    lenC = n
    c = x[lenA:]
    lenT = lenA + lenC
    row = 0
    col = 0
    i = 0
    while i < lenA:
#        print("row = {}, col = {}, i = {}".format(row, col, i))
        if row == col:
            gr[i] = (zi[row] - c[row])**2
        else:
            gr[i] = 2*(zi[row] - c[row])*(zi[col] - c[col])
        i += 1
        col += 1
        if col == n:
            row += 1
            col = row
    c_i = 0
    while i < lenT:
        gr[i] = -2*(zi-c).T@A[c_i, :]
        i += 1
        c_i += 1
    return gr
    

def jacobi(z, w, x, n):
    m = len(z)
    A, c = xTm(x, n)
    b = int(n*(n+1)/2)
    jac = np.zeros((m, b + n))
    
    for i in range(m):
        inside = (z[i] - c).T@A@(z[i] - c) <= 1
        if w[i] > 0:
            if inside:
                jac[i, :] = np.zeros((b+n, ))
            else:
                jac[i, :] = gradr(z[i], w, x, n)
        else:
            if inside:
                jac[i, :] = - gradr(z[i], w, x, n)
            else:
                jac[i, :] = np.zeros((b+n, ))
    return jac


def pseudograd(z, w, x, n, g, eps):
    res = f(z, w, x, n)
    fd = f(z, w, x-eps*g/np.linalg.norm(g, 2), n)
    return (fd - res)/eps
    

def grad(z, w, x, n):
    res = residuals(z, w, x, n)
    jac = jacobi(z, w, x, n)
    grad = 2*jac.T@res
    return grad


def backtracking_linesearch(z, w, x, n):
    g = grad(z, w, x, n)
    fi = f(z, w, x, n)
    alpha = 1
    while(f(z, w, x-alpha*g, n) > fi - 0.1*alpha*g.T@g):
        alpha *= 0.5
    return x - alpha*g

def steepest_descent_bt_search(z, w, x, n, TOL = 1e-5):
    print("f(x_0) = ", f(z, w, x, N, False))
    
    last = .0
    curr = f(z, w, x, N, False)
    iterations = 0
    
    while np.abs((curr - last)) > 1e-5:
        last = curr
        x = backtracking_linesearch(z, w, x, N)
        curr = f(z, w, x, N)
        iterations += 1
    
    print("f(x_{}) = {}".format(iterations, f(z, w, x, N)))
    
    return x


z, w = generate_points(10, 2)
color = [['green', 0, 'red'][1-i] for i in w] 
x = [1, 0.0, 1, .0, .0]

plt.scatter(z[:, 0], z[:, 1], c=color)
plot_ellipse(x, N)

g = grad(z, w, x, N)
p = pseudograd(z, w, x, N, g, 1e-5)

print("g = ", g)
print("g.T*g / ||g|| = ", -g.T@g/np.linalg.norm(g, 2))
print("Fin. Diff     = ", p)

x = steepest_descent_bt_search(z, w, x, N)

plot_ellipse(x, N)


