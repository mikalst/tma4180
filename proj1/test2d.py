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

def generate_points(num, dim=2):
    z = 2*np.random.rand(num, dim) - (1, 1)
    return z


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


def mTx(A, c, n):
    b = int(n*(n+1)/2)
    x = np.zeros((b+n, ))
    for i in range(n):
        for j in range(i, n):
            ind = b - int((n-i)*(n-i+1)/2) + j-i
            x[ind] = A[i, j]
    x[b:] = c
    return x


def residuals(z, w, x, n):
    m = len(z)
    res = np.zeros((m, ))
    A, c = xTm(x, n)
    for i in range(m):
        res[i] = ((z[i] - c).T@A@(z[i] - c)-1)*w[i]
        if (res[i] < 0):
            res[i] = 0
    return res


def f(z, w, x, n):
    return np.sum(residuals(z, w, x, n)**2)


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
        gr[i] = -2*A[c_i, :]@(zi-c)
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
    fd = f(z, w, x-eps*g.T@g, n)
    return (fd - res)/eps
    

def grad(z, w, x, n):
    res = residuals(z, w, x, n)
    jac = jacobi(z, w, x, n)
#    print("res = ", res,"jac = ", jac)
    grad = 2*jac.T@res
    return grad


def armijo_linesearch(z, w, x, n, g):
    g = grad(z, w, x, n)
    fi = f(z, w, x, n)
    alpha = 1
    while(f(z, w, x-alpha*g, n) > fi - 0.1*alpha*g.T@g):
        alpha *= 0.5
#    print(alpha)
    return x - alpha*g
    

z = generate_points(5, 2)
w = np.random.choice([-1, 1], 5)

color = [['green', 0, 'red'][1-i] for i in w] 
x = [1, 0.0, 1, .0, .0]
n = 2
plt.scatter(z[:, 0], z[:, 1], c=color)
plot_ellipse(x, n)

res = f(z, w, x, n)
g = grad(z, w, x, n)
x = armijo_linesearch(z, w, x, n, g)
print("Grad       = ", g)
print("Pseudograd = ", pseudograd(z, w, x, n, g, 1e-10))

for _ in range(20):
    x = armijo_linesearch(z, w, x, n, g)

plot_ellipse(x, n)


#Currently not working
#def bisection_linesearch(z, w, x_k, n, p_k, g):
#    fi = f(z, w, x_k, n)
#    c1 = 0.4
#    c2 = 0.8
#
#    alpha_min = 1
#    
#    sd = False
#    while(not sd):
#        alpha_min *= 0.5
#        sd = f(z, w, x_k + alpha_min * p_k, n) <= fi + c1 * alpha_min * g.T@p_k
#    
#    alpha_max = 2*alpha_min
#    alpha = (alpha_max + alpha_min)/2
#
#    sd = False
#    cc = False
#
#    while (not sd or not cc):
##        print(alpha)
#        sd = f(z, w, x_k + alpha_max * p_k, n) <= fi + c1*alpha_max*g.T@p_k
#        cc = grad(z, w, x_k + alpha * p_k, n).T@p_k >= c2*g.T@p_k
#        
#        if not sd:
#            alpha_max = alpha
#            
#        elif not cc:
#            alpha_min = alpha
#            
#        alpha = (alpha_max + alpha_min)/2
#        
#    return x_k + alpha*p_k
#    #Find point of no-sufficient decrease, as long as problem is coercive this
#    # will terminate
#    alpha_max = 1
#
#    while f(z, w, x_k + alpha_max * p_k, n) < fi + c1*alpha_max*g.T@p_k:
#        alpha_max *= 2
#
#    alpha_min = alpha_max
##    print(alpha_min)
#
#    while f(z, w, x_k + alpha_min * p_k, n) > fi + c1*alpha_min*g.T@p_k:
#        alpha_min /= 2
##        print(alpha_min)
#    
#    alpha = (alpha_max + alpha_min)/2
##    print(alpha)
#    
#    #Find point where wolfe-conditions are satisfied
#    while True:
#        if grad(z, w, x_k + alpha * p_k, n).T@p_k < c2*g.T@p_k:
#            alpha_min = alpha
#            alpha = (alpha_max + alpha_min) / 2
#        elif grad(z, w, x_k + alpha * p_k, n).T@p_k > -c2*g.T@p_k:
#            alpha_max = alpha
#            alpha = (alpha_max + alpha_min) / 2
#        else:
##            print(alpha)
#            return x_k + alpha * p_k
