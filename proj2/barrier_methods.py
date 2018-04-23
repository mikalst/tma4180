#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constrained_ellipsoids.py import *

# every constraint is a inequality constraint

def P(x, mu, con, z, w, f):
    P = f(z, w, x)
    for i in range(len(con)):
        P -= mu*con[i]
    return P

def grad_P(x, mu, con, con_gr, z, w):
    grad_P = jacobi(z, w, x)
    for i in range(len(con)):
        grad_P -= mu*con_gr[i]/con[i]
    return grad_P
     
def barrier(x0, mu0, f, TOL):
    k = 1
    x1 = x0 
    mu = mu0
    
    con = cf(x1)
    con_gr = cg(x)
    
    lambda p: P(x, mu, con, z, w, f)
    lambda g_p: grad_P(x, mu, con, con_gr, z, w)
    mu_vec = np.full(len(con), mu)
    
    while True:
        x1 = bfgs(p, g_p, TOL = 1/k) # have to modify bfgs to yield feasable points
        
        # lagrange multipliers
        z = np.divide(mu_vec, con(x1))
        
        # convergence test
        if mu < TOL:
            return x1
        
        # new penalty parameter
        mu = 1/2*mu
        
        k += 1

        if k > 10000:
            print('No convergence in {} steps'.format(k))
        
