#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:21:45 2018

@author: mikal
"""

import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import search_methods as sm


def generate_points(num, dim=2):
    z = 1.5*(2*np.random.rand(num, dim) - np.ones((dim, )))
    w = np.random.choice([-1, 1], size=num)
    return z, w


def xTm(x):
    """Converts the vector x to (A, c) in the following format
        A = x0, x1,   x2     ... xn-1
            x1, xn+1, xn+2 ... x2n-3
            x2, x1,   x2n-2  ... x3n-7
            
        c = (xn(n+1)/2, ... ,xn(n+1)/2 + n)
    """
    n = 2
    
    A = np.zeros((n, n))
    k = int(n*(n+1)/2)
    
    
    for i in range(n):
        for j in range(i, n):
            ind = k - int((n-i)*(n-i+1)/2) + j-i
            A[i, j] = x[ind]
            A[j, i] = x[ind]
    c = x[k:]# c can also be b but it requires no different implementation
    return A, c


def residual(z, w, x):
    A, c = xTm(x)
    r = (z.T@A@z + c.dot(z) -1)*w
    if r < 0:
        r = 0
    return r


def gradr(zi, w, x):
    n = 2
    A, c = xTm(x)
    gr = np.ones((len(x), ))
    lenA = int(n*(n+1)/2)
    lenC = n
    lenT = lenA + lenC
    row = 0
    col = 0
    i = 0

    while i < lenA:
        if row == col:
            gr[i] = (zi[row])**2
        else:
            gr[i] = 2*(zi[row])*(zi[col])
        i += 1
        col += 1
        if col == n:
            row += 1
            col = row
    c_i = 0
    while i < lenT:
        gr[i] = zi[c_i]
        i += 1
        c_i += 1
    return gr
        

def jacobi(z, w, x):
    n = 2
    m = len(z)
    A, c = xTm(x)
    b = int(n*(n+1)/2)
    jac = np.zeros((m, b + n))
    
    for i in range(m):
        inside = np.dot(z[i], np.dot(A, z[i])) + np.dot(c, z[i]) <= 1
        if w[i] > 0:
            if inside:
                jac[i, :] = np.zeros((b+n, ))
            else:
                jac[i, :] = gradr(z[i], w, x)
        else:
            if inside:
                jac[i, :] = - gradr(z[i], w, x)
            else:
                jac[i, :] = np.zeros((b+n, ))
    return jac


def residuals(z, w, x):
    res = np.zeros((len(z), ))
    for i in range(len(z)):
        res[i] = residual(z[i], w[i], x)
    return res


def F(z, w, x, printres=False):
    res = residuals(z, w, x)
    if printres: print(w); print(res)
    return np.sum(res**2)

    
def G(z, w, x):
    res = residuals(z, w, x)
    jac = jacobi(z, w, x)
    grad = 2*jac.T@res
    return grad


#Only implemented in 2D, as project 2 is only in 2D. 
def constraint_grad(x):
    num_constraints = 5
    g = np.zeros((num_constraints, len(x)))
    A, c = xTm(x)
    g[0, :] = np.array([A[0, 0], 0, 0, 0, 0])
    return


def setmodelzw(z, w, x):
    f = lambda x: F(z, w, x)
    g = lambda x: G(z, w, x)
    return f, g


def plot(x, z, pointcol, color='r', name='default'):
    A, c = xTm(x)
      
    points  = 150
    x = np.linspace(-1.5, 1.5, points)
    y = np.linspace(-1.5, 1.5, points)
    X, Y = np.meshgrid(x, y)
    V = np.ones((points, points))
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):   
            zi = np.array([X[i, j], Y[i, j]])
            V[i, j] = np.dot(zi, A).dot(zi) + c.dot(zi)
    
    CS = plt.contour(X, Y, V, levels=[1], colors=color)
    fmt = {}
    strs = [name]
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    plt.clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=10)
 

def test_grad():
    N = 2
    z, w = generate_points(10, 2)
    x = np.random.rand(5)
    f, g = setmodelzw(z, w, x)
    eps = [10**k for k in range(-1, -12, -1)]
    fi = f(x)
    gi = g(x)
#    
#    p = np.identity(int(2*(2+1)/2) + 2)
#    permute = np.random.rand(int(2*(2+1)/2) + 2, int(2*(2+1)/2) + 2)
#    p = permute@p
    
    p = np.random.rand((int(N*(N+1)/2 + N)))
    print("p = {}".format(p))
    for e in eps:
        print("ep = {:e}, error = {:e}".format(e, gi.dot(p)-(f(x + e*p) - fi)/e))         

#test_grad()


def test2D(rand=False):
    
    N = 2
    
    if rand:
        np.random.seed()
        seed = np.random.randint(100)
        np.random.seed(seed)
      
    nz = 10
    z, w = generate_points(nz, N)
    color = [['green', 0, 'red'][1-i] for i in w]
    x = np.random.rand(int(N*(N+1)/2 + N))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(111)

    f, g = setmodelzw(z, w, x)
   
    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
        
#    plot_ellipse(x, z, color, N, 'blue', 'Initial')
    plot(x1, z, color, 'orange', 'SD')
    plot(x2, z, color, 'purple', 'BFGS')
    plot(x3, z, color, 'yellow', 'FR')
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plt.title(r"SD=({}, {:.2f}), BFGS=({}, {:.2f}), FR=({}, {:.2f})".format(
            it1, err1, it2, err2, it3, err3), fontsize = 11.5)
    plt.show()
    
#test2D(True)



if __name__ == "__main__":
#    test_grad()
    test2D(True)

