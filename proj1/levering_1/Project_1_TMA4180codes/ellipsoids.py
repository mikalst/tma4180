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
from mpl_toolkits.mplot3d import Axes3D

import search_methods as sm

plt.style.use('ggplot')


def generate_points(num, dim=2):
    z = 1.5*(2*np.random.rand(num, dim) - np.ones((dim, )))
    w = np.random.choice([-1, 1], size=num)
    return z, w


def generate_square(num, dim=2, some_error = False):
    z = 1.5*(2*np.random.rand(num, dim) - np.ones((dim, )))
    w = np.zeros((num, ))
    for i, zz in enumerate(z):
#        print(zz)
        if np.max(np.abs(zz)) <= 1:
            w[i] = 1
        else:
            w[i] = -1
    for i, w_i in enumerate(w):
        if (np.random.rand() > 0.90):
            w[i] *= -1
            
#    print(w)
    return z, w


def xTm(x, n):
    A = np.zeros((n, n))
    k = int(n*(n+1)/2)
    for i in range(n):
        for j in range(i, n):
            ind = k - int((n-i)*(n-i+1)/2) + j-i
            A[i, j] = x[ind]
            A[j, i] = x[ind]
    c = x[k:]# c can also be b but it requires no different implementation
    return A, c


def residual(z, w, A, c, model):
    if model == 1:
        r = ((z - c).T@A@(z - c)-1)*w
    elif model == 2:
        r = (z.T@A@z + c.dot(z) -1)*w
    if r < 0:
        r = 0
    return r


def find_weights(zz, A, c, model, some_error = False):
    w = np.zeros(len(zz))
    for i, z in enumerate(zz):
        if model == 1:
            if np.dot(z-c, A).dot(z-c) <= 1:
                w[i] = 1
            else:
                w[i] = -1
        elif model == 2:
            if np.dot(z, A).dot(z) + c.dot(z) <= 1:
                w[i] = 1
            else:
                w[i] = -1
    return w


def gradr(zi, w, x, n, model):
    A, c = xTm(x, n)
    gr = np.ones((len(x), ))
    lenA = int(n*(n+1)/2)
    lenC = n
    c = x[lenA:]
    lenT = lenA + lenC
    row = 0
    col = 0
    i = 0
    if model == 1:
        
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
    
    elif model == 2:
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
        

def jacobi(z, w, x, n, model):
    m = len(z)
    A, c = xTm(x, n)
    b = int(n*(n+1)/2)
    jac = np.zeros((m, b + n))
    
    for i in range(m):
        if model == 1:
            inside = (z[i] - c).T@A@(z[i] - c) <= 1
        elif model == 2:
            inside = np.dot(z[i], np.dot(A, z[i])) + np.dot(c, z[i]) <= 1
        if w[i] > 0:
            if inside:
                jac[i, :] = np.zeros((b+n, ))
            else:
                jac[i, :] = gradr(z[i], w, x, n, model)
        else:
            if inside:
                jac[i, :] = - gradr(z[i], w, x, n, model)
            else:
                jac[i, :] = np.zeros((b+n, ))
    return jac


def residuals(z, w, x, n, model):
    res = np.zeros((len(z), ))
    A, c = xTm(x, n)
    for i in range(len(z)):
        res[i] = residual(z[i], w[i], A, c, model)
    return res


def F(z, w, x, n, model, printres=False):
    res = residuals(z, w, x, n, model)
    if printres: print(w); print(res)
    return np.sum(res**2)

    
def G(z, w, x, n, model):
    res = residuals(z, w, x, n, model)
    jac = jacobi(z, w, x, n, model)
    grad = 2*jac.T@res
    return grad


def setmodelzw(z, w, x, n, model):
    f = lambda x: F(z, w, x, n, model)
    g = lambda x: G(z, w, x, n, model)
    return f, g

def plot3D(x, z, pointcol, N, model, color='r', name='default'):
    A, c = xTm(x, N)
    
    points = 100
    X = np.linspace(-3.0, 3.0, points)
    Y = np.linspace(-3.0, 3.0, points)
    Z = np.linspace(-3.0, 3.0, points)
    
    indices = []
#        X, Y, Z = np.meshgrid(x, y, z)
#        print(x.shape, y.shape. z.shape)
    V = np.ones((points, points, points))
    for i in range(points):
        for j in range(points):
            for k in range(points):
                z_i = np.array([X[i], Y[j], Z[k]])
                V[i, j, k] = residual(z_i, 1, A, c, model)
                if np.abs(V[i, j, k] - 1) < 0.05:
                    indices.append([i, j, k])
    
    indices = np.array(indices)
    Xp = X[indices[:, 0]]
    Yp = Y[indices[:, 1]]
    Zp = Z[indices[:, 2]]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(xs=z[:,0], ys=z[:,1], zs=z[:, 2], s = 30, c = pointcol)
    ax.scatter(xs=Xp, ys=Yp, zs=Zp, s=0.5)
    plt.savefig("x={}.pdf".format(x))
    plt.show()

def plot(x, z, pointcol, N, model, color='r', name='default'):
    assert(N == 2)
    
    A, c = xTm(x, N)
      
    points  = 150
    x = np.linspace(-1.5, 1.5, points)
    y = np.linspace(-1.5, 1.5, points)
    X, Y = np.meshgrid(x, y)
    V = np.ones((points, points))
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):   
            zi = np.array([X[i, j], Y[i, j]])
            if model == 1:    
                V[i, j] = np.dot(zi-c, A).dot(zi - c)
            elif model == 2:
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
    f, g = setmodelzw(z, w, x, N, 2)
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
      
    nz = 30
    z, w = generate_points(nz, N)
    color = [['green', 0, 'red'][1-i] for i in w]
    x = np.random.rand(int(N*(N+1)/2 + N))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    model = 1

    f, g = setmodelzw(z, w, x, N, model)
   
    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
        
#    plot_ellipse(x, z, color, N, 'blue', 'Initial')
    plot(x1, z, color, N, model, 'orange', 'SD')
    plot(x2, z, color, N, model, 'purple', 'BFGS')
    plot(x3, z, color, N, model, 'yellow', 'FR')
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plt.title(r"SD=({}, {:.2f}), BFGS=({}, {:.2f}), FR=({}, {:.2f})".format(
            it1, err1, it2, err2, it3, err3), fontsize = 11.5)
    
    plt.subplot(122)
    model = 2
    f, g = setmodelzw(z, w, x, N, model)

#    x = np.array([1, 0., 0., 1., 0., 1. , 0., 0., 0.])
#    x = np.array([1, 0., 1, 0., 0.])   
            
    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
    
    plot(x1, z, color, N, model, 'orange', 'SD')
    plot(x2, z, color, N, model, 'purple', 'BFGS')
    plot(x3, z, color, N, model, 'yellow', 'FR')
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plt.title(r"SD=({}, {:.2f}), BFGS=({}, {:.2f}), FR=({}, {:.2f})".format(
            it1, err1, it2, err2, it3, err3), fontsize = 11.5)
    
    plt.savefig("Test_{}D_seed{}_nz{}.pdf".format(N, model, seed, nz))    
    plt.show()

#test2D(True)
    
def test3D(rand=False):
    
    N = 3
    
    if rand:
        np.random.seed()
        seed = np.random.randint(100)
        np.random.seed(seed)
      
    nz = 20
    z, w = generate_points(nz, N)
    color = [['green', 0, 'red'][1-i] for i in w]
    x = np.random.rand(int(N*(N+1)/2 + N))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    model = 1

    f, g = setmodelzw(z, w, x, N, model)
   
    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
        
#    plot_ellipse(x, z, color, N, 'blue', 'Initial')
    plot3D(x1, z, color, N, model, 'orange', 'SD')
    plot3D(x2, z, color, N, model, 'purple', 'BFGS')
    plot3D(x3, z, color, N, model, 'yellow', 'FR')

    model = 2
    f, g = setmodelzw(z, w, x, N, model)
            
    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
    
    plot3D(x1, z, color, N, model, 'orange', 'SD')
    plot3D(x2, z, color, N, model, 'purple', 'BFGS')
    plot3D(x3, z, color, N, model, 'yellow', 'FR')

    
def test_problem1():
    
    np.random.seed()
    N = 2
    
    case = {0: 'circle',1: 'ellipse'}[1]
    
    if case == 'circle':
        x0 = np.array([1., 0., 1., 0.5, 0.5])    
    if case == 'ellipse':
        x0 = np.array([1., 0., 1E2, 0.0, 0.0])    

    A, c = xTm(x0, N)
    z, w = generate_points(150, N)
        
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    model = 1

    w = find_weights(z, A, c, model)
    color = [['green', 0, 'red'][int(1-i)] for i in w]
    x = 2*(1-np.random.rand(5))
    f, g = setmodelzw(z, w, x, N, model)
    
    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
    
    plot(x0, z, color, N, model, 'blue', 'Real')        
    plot(x1, z, color, N, model, 'orange', 'SD')
    plot(x2, z, color, N, model, 'purple', 'BFGS')
    plot(x3, z, color, N, model, 'yellow', 'FR')
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plt.title(r"SD=({}, {:.2f}), BFGS=({}, {:.2f}), FR=({}, {:.2f})".format(
            it1, err1, it2, err2, it3, err3), fontsize = 11.5)
    
    plt.subplot(122)
    model = 2
    w = find_weights(z, A, c, model)
    color = [['green', 0, 'red'][int(1-i)] for i in w]
    x = 2*(1-np.random.rand(5))
    f, g = setmodelzw(z, w, x, N, model)


    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
    
    plot(x0, z, color, N, model, 'blue', 'Real') 
    plot(x1, z, color, N, model, 'orange', 'SD')
    plot(x2, z, color, N, model, 'purple', 'BFGS')
    plot(x3, z, color, N, model, 'yellow', 'FR')
    plt.scatter(z[:, 0], z[:, 1], c=color)    
    plt.title(r"SD=({}, {:.2f}), BFGS=({}, {:.2f}), FR=({}, {:.2f})".format(
            it1, err1, it2, err2, it3, err3), fontsize = 11.5)
    
    plt.savefig("{}.pdf".format(case))    
    plt.show()


#test_problem1()  
       

def test_problem2():

    N = 2
    np.random.seed()

    #Unit square
    z, w = generate_square(200, N, False)
    x = 2*(1-np.random.rand(int(N*(N+1)/2 + N)))
    color = [['green', 0, 'red'][int(1-i)] for i in w]

        
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    model = 1

    color = [['green', 0, 'red'][int(1-i)] for i in w]
    x = 2*(1-np.random.rand(int(N*(N+1)/2 + N)))
    f, g = setmodelzw(z, w, x, N, model)
    
    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
    
    plot(x1, z, color, N, model, 'orange', 'SD')
    plot(x2, z, color, N, model, 'purple', 'BFGS')
    plot(x3, z, color, N, model, 'yellow', 'FR')
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plt.title(r"SD=({}, {:.2f}), BFGS=({}, {:.2f}), FR=({}, {:.2f})".format(
            it1, err1, it2, err2, it3, err3), fontsize = 11.5)
    
    plt.subplot(122)
    model = 2
    x = 2*(1-np.random.rand(int(N*(N+1)/2 + N)))
    f, g = setmodelzw(z, w, x, N, model)


    x1, it1, err1 = sm.steepest_descent(f, g, x)
    x2, it2, err2 = sm.bfgs(f, g, x)
    x3, it3, err3 = sm.fletcher_reeves(f, g, x)
    
    plot(x1, z, color, N, model, 'orange', 'SD')
    plot(x2, z, color, N, model, 'purple', 'BFGS')
    plot(x3, z, color, N, model, 'yellow', 'FR')
    plt.scatter(z[:, 0], z[:, 1], c=color)    
    plt.title(r"SD=({}, {:.2f}), BFGS=({}, {:.2f}), FR=({}, {:.2f})".format(
            it1, err1, it2, err2, it3, err3), fontsize = 11.5)
    
    plt.savefig("Square.pdf")
    plt.show()

        

if __name__ == "__main__":
    test_grad()
    test2D(True)
    test3D(True)
    test_problem1() 
    test_problem2()

    
