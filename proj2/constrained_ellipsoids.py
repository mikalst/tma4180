#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:21:45 2018

@author: mikal
"""

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from scipy.optimize import minimize as minimize

import search_methods as sm
#import constrained_search_methods as csm
import barrier_methods as bm


def find_weights(zz, A, c):
    w = np.zeros(len(zz), dtype = np.int)
    for i, z in enumerate(zz):
        if np.dot(z, A).dot(z) + c.dot(z) <= 1:
            w[i] = 1
        else:
            w[i] = -1
    return w


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


def testScipy():
    nz = 500
    N = 2
    z, w = generate_points(nz, N)
    
    A = np.array([[1, -10], [-10, 1]])
    c = np.array([.0, .0])
    w = find_weights(z, A, c)
    
    x = np.random.rand(int(N*(N+1)/2 + N))
    color = [['green', 0, 'red'][1-i] for i in w]
    
    f, g = setmodelzw(z, w, x)
    
    lambda_l = 1
    lambda_h = 1E3
    
    constraint1 = {'type': 'ineq',
                   'fun': lambda x: x[0] - lambda_l,
                   'jac': lambda x: np.array([1, 0, 0, 0, 0])}
    
    constraint2 = {'type': 'ineq',
                   'fun': lambda x: -x[0] + lambda_h,
                   'jac': lambda x: np.array([-1, 0, 0, 0, 0])}
    
    constraint3 = {'type': 'ineq',
                   'fun': lambda x: x[2] - lambda_l,
                   'jac': lambda x: np.array([0, 0, 1, 0, 0])}
    
    constraint4 = {'type': 'ineq',
                   'fun': lambda x: -x[2] + lambda_h,
                   'jac': lambda x: np.array([0, 0, -1, 0, 0])}
    
    constraint5 = {'type': 'ineq',
                   'fun': lambda x: np.power(x[0]*x[2], 0.5) - np.power(lambda_l**2 + x[1]**2, 0.5),
                   'jac': lambda x: np.array([0.5*x[2]*np.power(x[0]*x[2], -0.5),
                                              -np.power(lambda_l**2 + x[1]**2, -0.5)*x[1],
                                              0.5*x[0]*np.power(x[0]*x[2], -0.5),
                                              0,
                                              0])}
    
    def cf(x):
        CONSTRAINTS = 5
        cc = np.zeros((CONSTRAINTS, ))
        cc[0] = x[0] - lambda_l
        cc[1] = -x[0] + lambda_h
        cc[2] = x[2] - lambda_l
        cc[3] = -x[2] + lambda_h
        cc[4] = np.power(x[0]*x[2], 0.5) - np.power(lambda_l**2 + x[1]**2, 0.5)
        return cc
    
    def cg(x):
        CONSTRAINTS = 5
        cg = np.zeros((CONSTRAINTS, CONSTRAINTS))
        cg[:, 0] = np.array([1, 0, 0, 0, 0])
        cg[:, 1] = np.array([-1, 0, 0, 0, 0])
        cg[:, 2] = np.array([0, 0, 1, 0, 0])
        cg[:, 3] = np.array([0, 0, -1, 0, 0])
        cg[:, 4] = np.array([0.5*x[2]*np.power(x[0]*x[2], -0.5), -np.power(lambda_l**2 + x[1]**2, -0.5)*x[1],
                                              0.5*x[0]*np.power(x[0]*x[2], -0.5), 0, 0])
        return cg
        
    
    x0 = minimize(f, x, jac=g, method = 'SLSQP', constraints = [constraint1,
                                                                constraint2,
                                                                constraint3,
                                                                constraint4,
                                                                constraint5])
    
#    x1 = minimize(f, x, jac=g, method = 'COBYLA', constraints = [constraint1,
#                                                                constraint2,
#                                                                constraint3,
#                                                                constraint4,
#                                                                constraint5])
    
    l_0 = np.zeros(5)
    csm.linesearch_sqp(x, l_0, f, g, cf, cg)
    
    x2, it2, err2 = sm.bfgs(f, g, x)
    
    print(x0)
    
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plot(x0.x, z, color, 'orange', 'SLSQP')
#    plot(x1.x, z, color, 'purple', 'COBYLA')
    plot(x2, z, color, 'yellow', 'BFGS - Unconstrained')
    
    #Alltid lettere med likhet, problemet er at 
    #
    #Barrier / Augmented Lagrangian er kanskje enklere. 
    
  
def cf(x, l, h):
    CONSTRAINTS = 5
    cc = np.zeros((CONSTRAINTS, ))
    cc[0] = x[0] - l
    cc[1] = -x[0] + h
    cc[2] = x[2] - l
    cc[3] = -x[2] + h
    cc[4] = np.power(x[0]*x[2], 0.5) - np.power(l**2 + x[1]**2, 0.5)
    return cc
    
def cg(x, l):
    CONSTRAINTS = 5
    cg = np.zeros((CONSTRAINTS, CONSTRAINTS))
    cg[:, 0] = np.array([1, 0, 0, 0, 0])
    cg[:, 1] = np.array([-1, 0, 0, 0, 0])
    cg[:, 2] = np.array([0, 0, 1, 0, 0])
    cg[:, 3] = np.array([0, 0, -1, 0, 0])
    cg[:, 4] = np.array([0.5*x[2]*np.power(x[0]*x[2], -0.5), -np.power(l**2 + x[1]**2, -0.5)*x[1],
                                          0.5*x[0]*np.power(x[0]*x[2], -0.5), 0, 0])
    return cg



def test_barrier():
    nz = 500
    N = 2
    z, w = generate_points(nz, N)
    mu0 = 1
    
    A = np.array([[1, -10], [-10, 1]])
    c = np.array([.0, .0])
    w = find_weights(z, A, c)
    
    x = np.random.rand(int(N*(N+1)/2 + N))
    #color = [['green', 0, 'red'][1-i] for i in w]
    
    
    lambda_l = 1
    lambda_h = 1E3
    
    constraint = lambda x: cf(x, lambda_l, lambda_h)
    constraint_grad = lambda x: cg(x, lambda_l)
    
    f, g = setmodelzw(z, w, x)
    
    x = bm.barrier(x, mu0, constraint, constraint_grad, lambda_l, f, g)
    
    return x



if __name__ == "__main__":
#    test_grad()
#    testScipy()
#    test2D(True)
    test_barrier()
