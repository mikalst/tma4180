#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:21:45 2018

@author: mikal
"""

import numpy as np
import matplotlib.pyplot as plt
import search_methods as sm
#import sqp_methods as sqp
import barrier_methods as bm


def generate_points(num):
    z = 1.5*(2*np.random.rand(num, 2) - np.ones((2, )))
    w = np.random.choice([-1, 1], size=num)    
    return z, w


def find_weights(zz, x, some_error = False):
    A, c = xTm(x)
    w = np.zeros(len(zz), dtype = np.int)
    for i, z in enumerate(zz):
        if np.dot(z, A).dot(z) + c.dot(z) <= 1:
            w[i] = 1
        else:
            w[i] = -1
    
    if some_error:
        for i, w_i in enumerate(w):
            if (np.random.rand() > 0.90):
                w[i] *= -1
            
    return w


def generate_square(num, some_error = False):
    z = 1.5*(2*np.random.rand(num, 2) - np.ones((2, )))
    w = np.zeros((num, ), dtype=np.int)
    for i, zz in enumerate(z):
        if np.max(np.abs(zz)) <= 1:
            w[i] = 1
        else:
            w[i] = -1
    if some_error:
        for i, w_i in enumerate(w):
            if (np.random.rand() > 0.80):
                w[i] *= -1
            
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
    c = x[k:]
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


def constraint_grad(x):
    num_constraints = 5
    g = np.zeros((num_constraints, len(x)))
    A, c = xTm(x)
    g[0, :] = np.array([A[0, 0], 0, 0, 0, 0])
    return 


def setmodelzw(z, w):
    f = lambda x: F(z, w, x)
    g = lambda x: G(z, w, x)
    return f, g


def plot(x, z, pointcol, color='r', name='default', alpha = 1):
    A, c = xTm(x)
      
    points  = 150
    x = np.linspace(-1.5, 1.5, points)
    y = np.linspace(-1.5, 1.5, points)
    X, Y = np.meshgrid(x, y)
    V = np.zeros((points, points))
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):   
            zi = np.array([X[i, j], Y[i, j]])
            V[i, j] = np.dot(zi, A).dot(zi) + c.dot(zi)
    
    CS = plt.contour(X, Y, V, levels=[1], colors=color, alpha = alpha)
    fmt = {}
    strs = [name]
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    plt.clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)


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
    cg[0, :] = np.array([1, 0, 0, 0, 0])
    cg[1, :] = np.array([-1, 0, 0, 0, 0])
    cg[2, :] = np.array([0, 0, 1, 0, 0])
    cg[3, :] = np.array([0, 0, -1, 0, 0])
    cg[4, :] = np.array([0.5*x[2]*np.power(x[0]*x[2], -0.5), -np.power(l**2 + x[1]**2, -0.5)*x[1],
                                          0.5*x[0]*np.power(x[0]*x[2], -0.5), 0, 0])
    return cg


def test_finite_difference_constraints():
    np.random.seed()
    testx = 1+np.random.randn(5)
    testp = np.random.randn(5)
    
    ll = 1E3 * np.random.rand()
    lh = ll + 1E6 * np.random.rand()
    
    print('ll = {:.2E}, lh = {:.2E}'.format(ll, lh))
    maxval = -np.inf
    for eps in np.logspace(0, -9, 10):
        err = max((cf(testx + eps*testp, ll, lh) - cf(testx, ll, lh))/(eps) - cg(testx, ll).dot(testp))
        maxval = max(err, maxval)
        print("Eps = {:.2E}".format(eps), '{:.5E}'.format(err))
    if (maxval <= 1E-1):
        print("===> F and G match")


#def scipy_constraints(lambda_l, lambda_h):
#    constraint1 = {'type': 'ineq',
#                   'fun': lambda x: x[0] - lambda_l,
#                   'jac': lambda x: np.array([1, 0, 0, 0, 0])}
#    
#    constraint2 = {'type': 'ineq',
#                   'fun': lambda x: -x[0] + lambda_h,
#                   'jac': lambda x: np.array([-1, 0, 0, 0, 0])}
#    
#    constraint3 = {'type': 'ineq',
#                   'fun': lambda x: x[2] - lambda_l,
#                   'jac': lambda x: np.array([0, 0, 1, 0, 0])}
#    
#    constraint4 = {'type': 'ineq',
#                   'fun': lambda x: -x[2] + lambda_h,
#                   'jac': lambda x: np.array([0, 0, -1, 0, 0])}
#    
#    constraint5 = {'type': 'ineq',
#                   'fun': lambda x: np.power(x[0]*x[2], 0.5) - np.power(lambda_l**2 + x[1]**2, 0.5),
#                   'jac': lambda x: np.array([0.5*x[2]*np.power(x[0]*x[2], -0.5),
#                                              -np.power(lambda_l**2 + x[1]**2, -0.5)*x[1],
#                                              0.5*x[0]*np.power(x[0]*x[2], -0.5),
#                                              0,
#                                              0])}       
#    return [constraint1, constraint2, constraint3, constraint4, constraint5]


def test_barrier():
    nz = 80
    z, w = generate_points(nz)
    mu0 = 2
    
    color = [['green', 0, 'red'][1-i] for i in w]
    
    #Choose random lambda to test robustness of algorithm.
    lambda_l = np.exp(-6 + 12*np.random.rand())
    lambda_h = lambda_l + np.exp(-6 + 12*np.random.rand())
    
#    If the span becomes really small we do not get convergence.
    lambda_l = 8E-1
    lambda_h = 1E2

    print("Running barrier with,\n", 
          "ll = {:.2E},\nlh = {:.2E},\n".format(lambda_l, lambda_h),
          "nz = {}".format(nz), sep = "")
    
    #Reduce function, gradient, constraint vector and constraint jacobi matrix
    # to 1 variable for call simplicity
    constraint =      lambda x: cf(x, lambda_l, lambda_h)
    constraint_grad = lambda x: cg(x, lambda_l)
    f, g = setmodelzw(z, w)
    
    #Initialize with feasible initial point
    x = np.zeros(5)
    x[0] = (lambda_l + lambda_h)/2
    x[1] = 0
    x[2] = (lambda_l + lambda_h)/2
    print("Found initial feasible point")
    
    #Call barriers
    x1, it1, err1 = bm.barrier(x, mu0, constraint, constraint_grad, lambda_l, f, g)  

    #Compare with unconstrained solution, BFGS
    x3_unconstrained, it3, err3 = sm.bfgs(f, g, x, max_iter = 1000)
    
    plt.title(r"$\lambda = [{:.2E}, {:.2E}]$, LB=({}, {:.2E})".format(lambda_l, lambda_h, it1, f(x1)), fontsize = 10)
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plot(x1, z, color, 'purple', 'Barrier')
    plot(x3_unconstrained, z, color, 'blue', 'BFGS')
    plt.savefig("figures/randompoints_ll{:.2E}_lh{:.2E}.pdf".format(lambda_l, lambda_h))
    plt.show()
    
    lambda_l = 1E1
    lambda_h = 1E2
    
    constraint =      lambda x: cf(x, lambda_l, lambda_h)
    constraint_grad = lambda x: cg(x, lambda_l)
    f, g = setmodelzw(z, w)
    
    #Initialize with feasible initial point
    x = np.zeros(5)
    x[0] = (lambda_l + lambda_h)/2
    x[1] = 0
    x[2] = (lambda_l + lambda_h)/2
    print("Found initial feasible point")
    
    #Call barriers
    x1, it1, err1 = bm.barrier(x, mu0, constraint, constraint_grad, lambda_l, f, g)  

    #Compare with unconstrained solution, BFGS
    x3_unconstrained, it3, err3 = sm.bfgs(f, g, x, max_iter = 1000)
    
    plt.title(r"$\lambda = [{:.2E}, {:.2E}]$, LB=({}, {:.2E})".format(lambda_l, lambda_h, it1, f(x1)), fontsize = 10)
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plot(x1, z, color, 'purple', 'Barrier')
    plot(x3_unconstrained, z, color, 'blue', 'BFGS')
    plt.savefig("figures/randompoints_ll{:.2E}_lh{:.2E}.pdf".format(lambda_l, lambda_h))

    
    return x


def barrier_ellipsoidal_set():
    nz = 40
    z, w = generate_points(nz)
    mu0 = 2
    
    #Ellipsoidal set
    x0 = np.array([1, -10, 1, 0, 0])
    print(x0)
    print("Found set")
    w = find_weights(z, x0, False)
    
    #Set color of points
    color = [['green', 0, 'red'][1-i] for i in w]
    
    #Specify lambda
    lambda_l = 1E-4
    lambda_h = 1E4

    print("Running barrier with,\n", 
          "ll = {:.2E},\nlh = {:.2E},\n".format(lambda_l, lambda_h),
          "nz = {}".format(nz), sep = "")
    
    #Reduce function, gradient, constraint vector and constraint jacobi matrix
    # to 1 variable for call simplicity
    constraint =      lambda x: cf(x, lambda_l, lambda_h)
    constraint_grad = lambda x: cg(x, lambda_l)
    f, g = setmodelzw(z, w)
    
    #Initialize with feasible initial point
    x = np.zeros(5)
    x[0] = (lambda_l + lambda_h)/2
    x[1] = 0
    x[2] = (lambda_l + lambda_h)/2
    print("Found initial feasible point")
    
    #Call barriers
    x1, it1, err1 = bm.barrier(x, mu0, constraint, constraint_grad, lambda_l, f, g)  

    #Compare with unconstrained solution, BFGS
    x3_unconstrained, it3, err3 = sm.bfgs(f, g, x, max_iter = 1000)
    
    plt.title(r"$\lambda = [{:.2E}, {:.2E}]$, LB=({}, {:.2E})".format(lambda_l, lambda_h, it1, f(x1)), fontsize = 10)
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plot(x1, z, color, 'purple', 'Barrier')
    plot(x0, z, color, 'orange', 'Real')
    plot(x3_unconstrained, z, color, 'blue', 'BFGS')
    plt.savefig("figures/randompoints_ll{:.2E}_lh{:.2E}.pdf".format(lambda_l, lambda_h))
    plt.show()
    
    lambda_l = 8E-1
    lambda_h = 1E2
    
    constraint =      lambda x: cf(x, lambda_l, lambda_h)
    constraint_grad = lambda x: cg(x, lambda_l)
    f, g = setmodelzw(z, w)
    
    #Initialize with feasible initial point
    x = np.zeros(5)
    x[0] = (lambda_l + lambda_h)/2
    x[1] = 0
    x[2] = (lambda_l + lambda_h)/2
    print("Found initial feasible point")
    
    #Call barriers
    x1, it1, err1 = bm.barrier(x, mu0, constraint, constraint_grad, lambda_l, f, g)  

    #Compare with unconstrained solution, BFGS
    x3_unconstrained, it3, err3 = sm.bfgs(f, g, x, max_iter = 1000)
    
    plt.title(r"$\lambda = [{:.2E}, {:.2E}]$, LB=({}, {:.2E})".format(lambda_l, lambda_h, it1, f(x1)), fontsize = 10)
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plot(x1, z, color, 'purple', 'Barrier')
    plot(x0, z, color, 'orange', 'Real')
    plot(x3_unconstrained, z, color, 'blue', 'BFGS')
    plt.savefig("figures/randompoints_ll{:.2E}_lh{:.2E}.pdf".format(lambda_l, lambda_h))
    
    return x


def barrier_square_set():
    
    #Specify lambda
    lambda_l = 1E10
    lambda_h = 1E11
    
    print("Running barrier with,\n", 
          "ll = {:.2E},\nlh = {:.2E},\n".format(lambda_l, lambda_h), sep = "")
    
    #Reduce constraint vector and constraint jacobi matrix to 1 variable
    constraint =      lambda x: cf(x, lambda_l, lambda_h)
    constraint_grad = lambda x: cg(x, lambda_l)
    
    nz = 40
#    mu0 = 2
    mu0 = 1E-8
    np.random.seed(42)
    z, w = generate_square(nz)
    
    #Set color of points
    color = [['green', 0, 'red'][1-i] for i in w]
    
    #Reduce objective and gradient to 1 variable
    f, g = setmodelzw(z, w)
    
    #Initialize with feasible initial point
    x = np.zeros(5)
    x[0] = (lambda_l + lambda_h)/2
    x[1] = 0
    x[2] = (lambda_l + lambda_h)/2
    print("Found initial feasible point")
    
    #Call barrier
    x1, it1, err1 = bm.barrier(x, mu0, constraint, constraint_grad, lambda_l, f, g)  

    #Compare with unconstrained solution, BFGS
    x3_unconstrained, it3, err3 = sm.bfgs(f, g, x, max_iter = 1000)
    
    plt.title(r"$\lambda = [{:.2E}, {:.2E}]$, LB=({}, {:.2E})".format(lambda_l, lambda_h, it1, f(x1)), fontsize = 10)
    plt.scatter(z[:, 0], z[:, 1], c=color)
    plot(x1, z, color, 'purple', 'Barrier')
    plot(x3_unconstrained, z, color, 'blue', 'BFGS')
    plt.plot([-1, -1], [-1, 1], [-1, 1], [1, 1], [1, 1], [-1, 1], [-1, 1], [-1, -1], color = "orange")
    plt.savefig("figures/randomset_ll{:.2E}_lh{:.2E}.pdf".format(lambda_l, lambda_h))
    plt.show()
    
#    lambda_l = 1E2
#    lambda_h = 1E3
#    
#    constraint =      lambda x: cf(x, lambda_l, lambda_h)
#    constraint_grad = lambda x: cg(x, lambda_l)
#    f, g = setmodelzw(z, w)
#    
#    #Initialize with feasible initial point
#    # below
#    x = np.zeros(5)
#    x[0] = (lambda_l + lambda_h)/2
#    x[1] = 0
#    x[2] = (lambda_l + lambda_h)/2
#    print("Found initial feasible point")
#    
#    #Call barriers
#    x1, it1, err1 = bm.barrier(x, mu0, constraint, constraint_grad, lambda_l, f, g, TOL = 1E-5)  
#
#    #Compare with unconstrained solution, BFGS
#    x3_unconstrained, it3, err3 = sm.bfgs(f, g, x, max_iter = 1000)
#    
#    plt.title(r"$\lambda = [{:.2E}, {:.2E}]$, LB=({}, {:.2E})".format(lambda_l, lambda_h, it1, f(x1)), fontsize = 10)
#    plt.scatter(z[:, 0], z[:, 1], c=color)
#    plot(x1, z, color, 'purple', 'Barrier')
#    plot(x3_unconstrained, z, color, 'blue', 'BFGS')
#    plt.plot([-1, -1], [-1, 1], [-1, 1], [1, 1], [1, 1], [-1, 1], [-1, 1], [-1, -1], color = "orange")
#    plt.savefig("figures/randompoints_ll{:.2E}_lh{:.2E}.pdf".format(lambda_l, lambda_h))
#    plt.show()
    
    return x




if __name__ == "__main__":
#    test_barrier()
#    barrier_ellipsoidal_set()
    barrier_square_set()
#    test_finite_difference_constraints()
