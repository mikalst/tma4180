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
    b = x[b:]
    return A, b


#def plot_ellipse(x, n):
#    A, b = xTm(x, n)
#    Ai = np.linalg.inv(A)
#    w, v = np.linalg.eig(A)
#    t = np.linspace(0, 2*np.pi, 100)
#    z_star = np.reshape([np.cos(t), np.sin(t)], (2, 100)).T
#    z = z_star@v/np.sqrt(w) + A
#    plt.plot(z[:, 0], z[:, 1])        


def residuals(z, w, x, n):
    m = len(z)
    res = np.zeros((m, ))
    A, b = xTm(x, n)
    for i in range(m):
        res[i] = (np.dot(z[i], np.dot(A, z[i])) + np.dot(b, z[i]) - 1)*w[i]
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
    b = x[lenA:]
    lenT = lenA + lenC
    row = 0
    col = 0
    i = 0
    while i < lenA:
#        print("row = {}, col = {}, i = {}".format(row, col, i))
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
    

def jacobi(z, w, x, n):
    m = len(z)
    A, b = xTm(x, n)
    k = int(n*(n+1)/2)
    jac = np.zeros((m, k + n))
    
    for i in range(m):
        inside = np.dot(z[i], np.dot(A, z[i])) + np.dot(b, z[i]) <= 1
        if w[i] > 0:
            if inside:
                jac[i, :] = np.zeros((k+n, ))
            else:
                jac[i, :] = gradr(z[i], w, x, n)
        else:
            if inside:
                jac[i, :] = - gradr(z[i], w, x, n)
            else:
                jac[i, :] = np.zeros((k+n, ))
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


def test_grad():
    z, w = generate_points(10, 2)
    x = [1, 0.0, 1, .0, .0]
    eps = 1e-8
    fi = f(z, w, x, 2)
    g = grad(z, w, x, 2)
    
    p = np.identity(int(2*(2+1)/2) + 2)
    permute = np.random.rand(int(2*(2+1)/2) + 2, int(2*(2+1)/2) + 2)
    p = permute@p

    print("Random directions")    
    for direction in p[:]:
        print("p =          {}".format(direction))
        print("Grad - Pseudograd = {}".format(g.dot(direction)-(f(z, w, x + eps*direction, 2) - fi)/eps))        
    print("Steepest descent")
    print("Grad =       {}".format(g.dot(-g)))
    print("Pseudograd = {}".format((f(z, w, x - eps*g, 2) - fi)/eps))
    
test_grad()


def backtracking_linesearch(z, w, x_k, n, p_k, g):
    fi = f(z, w, x_k, n)
    alpha = 1
    c1 = 0.2

    sd = False
    while(not sd):
        alpha *= 0.5
        sd = f(z, w, x_k + alpha * p_k, n) <= fi + c1 * alpha * g.T@p_k
        
    return x_k + alpha*p_k



#def bisection_linesearch(z, w, x, n, p, g):
#    c1 = 0.20
#    c2 = 0.5
#    alpha_min = 0.0
#    alpha_max = np.inf
#    alpha = 1
#    fx   = f(z, w, x, n)
#    g    = grad(z, w, x, n)
#    gxp  = g.T@p
#    while alpha < 1.0E+10:
##        print(alpha,alpha_min,alpha_max)
#        if f(z, w, x+alpha*p, n) > fx + alpha*c1*gxp:
#            # no sufficient decrease - too long step
##            print(g.T@p_k, f(z, w, x + alpha*p_k, n), fx + alpha*c1*gxp)
#            alpha_max = alpha
#            alpha = 0.5*(alpha_max + alpha_min)
#        elif grad(z, w, x+alpha*p, n).dot(p) < c2*gxp:
#            # no curvature condition: too short step
#            alpha_min = alpha
#            if alpha_max == np.inf:
#                alpha *= 2.0
#            else:
#                alpha = 0.5*(alpha_max + alpha_min)
#        else:
#            # we are done!
##            print(alpha)
#            return x + alpha * p
#    
#    raise ValueError('Steplength is way too long!')


def bisection_linesearch(z, w, x, n, p, g):
    alpha0 = 1E0
    c1 = 0.10
    c2 = 0.7
    
    alpha_min, alpha_max = 0.0, np.inf
    alpha = alpha0
    fx   = f(z, w, x, n)
    dfx  = grad(z, w, x, n)
    dfxp = dfx.dot(p)
    
    while alpha < 1.0E+10:
        #print(alpha,alpha_min,alpha_max)
        if f(z, w, x+alpha*p, n) > fx + alpha*c1*dfxp:
            # no sufficient decrease - too long step
            alpha_max = alpha
            alpha = 0.5*(alpha_max + alpha_min)
        elif grad(z, w, x+alpha*p, n).dot(p) < c2*dfxp:
            # no curvature condition: too short step
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha *= 2.0
            else:
                alpha = 0.5*(alpha_max + alpha_min)
        else:
            # we are done!
            return x + alpha * p



def steepest_descent(z, w, x, n, TOL = 1e-2):
    print("SD \nf(x_0) = ", f(z, w, x, N, False))

    g = grad(z, w, x, n)
    if np.linalg.norm(g) < TOL:
        return x
    
    p_k = -g

    iterations = 0

    while True:
        x = backtracking_linesearch(z, w, x, N, p_k, g)
        g = grad(z, w, x, n)
        
        iterations += 1
        if np.linalg.norm(g) < TOL or iterations >= 10000:
            break
        
        p_k = -g
    
    print("f(x_{}) = {}".format(iterations, f(z, w, x, N)))
    
    return x


def bfgs(z, w, x, n, TOL = 1e-2):
    print("BFGS \nf(x_0) = ", f(z, w, x, N, False))

    I = np.identity(int(n*(n+1)/2 + n))
    iterations = 0
    g = grad(z, w, x, n)
    H = 0.5 * I / np.linalg.norm(g)
    
    while True:
        x0 = x
        g0 = g
        
        p = - H@g
        
        x = bisection_linesearch(z, w, x, N, p, g)# This does satisfy wolfe-conditionns
        g = grad(z, w, x, n)
        
        iterations += 1
        if np.linalg.norm(g) < TOL or iterations >= 10000:
            break
 
        s = x - x0
        y = g - g0
        #For some reason y.dot(s) becomes 0 at unsavoury moments, which means that y and s are orthogonal
        # what is the analytical implication of this?
        rho = 1/y.dot(s); #print(y.dot(s))
        H = (I - rho*np.outer(s, y))@H@(I - rho*np.outer(y, s)) + rho*np.outer(s, s)
    
    print("f(x_{}) = {}".format(iterations, f(z, w, x, N)))
    
    return x


#def dfp(z, w, x, n, TOL=5e-2):
#    print("f(x_0) = ", f(z, w, x, N, False))
#    
#    I = np.identity(int(n*(n+1)/2 + n))
#    g = grad(z, w, x, n)
#    H = 0.1 * I / np.linalg.norm(g)
#    p = - H@g
#    
#    iterations = 0
#    while np.linalg.norm(g, 2) > TOL:
        



#List of difficult seeds
#np.random.seed(55)
np.random.seed(64)
#On this one, SD user 10000 while BFGS only 25
#np.random.seed(73)    

z, w = generate_points(7, 2)
color = [['green', 0, 'red'][1-i] for i in w] 
x = [1, 0.0, 1, .0, .0]

#g = grad(z, w, x, N)
#p = pseudograd(z, w, x, N, g, 1e-5)
#print("g = ", g)
#print("g.T*g / ||g|| = ", -g.T@g/np.linalg.norm(g, 2))

#x1 = steepest_descent(z, w, x, N)
x2 = bfgs(z, w, x, N)

#plot_ellipse(x, N)
#plot_ellipse(x1, N)
#plot_ellipse(x2, N)
plt.scatter(z[:, 0], z[:, 1], c=color)
plt.legend(["Initial", "SD", "BFGS"])