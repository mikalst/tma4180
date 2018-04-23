#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import search_methods as sm


# every constraint is a inequality constraint

def P(x, mu, con, f):
    assert((con > 0).all())
    P = f(x) - mu*sum(np.log(con))
    return P

def grad_P(x, mu, con, con_gr, g):
    con = con.reshape((len(con), ))
    grad_P = g(x) - mu*np.sum(con_gr/con, axis = 0)
    return grad_P


def check_KKT(x, mult, l, cf, g):
    TOL1 = 10E-3
    TOL2 = 10E-3
    
    if np.linalg.norm(g(x) - np.array((mult[0] - mult[1] + mult[4]/2*np.sqrt(x[2]/x[1]),
                             mult[2] - mult[3] + mult[4]/2*np.sqrt(x[0]/x[2]), 
                             mult[3]*x[1]/np.sqrt(l**2 + x[1]**2))), 2) > TOL1:
        return False
    con = cf(x)
    for c in con:
        if con < 0:
            return False
    for m in mult:
        if m < 0:
            return False
    for i in range(mult):
        if mult[i]*con[i] > TOL2:
            return False
    return True
        

def barrier(x0, mu0, cf, cg, l_eigen, f, g):
    k = 1
    mu = mu0
    x1 = x0
    
    con = cf(x1)
    con_gr = cg(x1)
    
    p = lambda x: P(x, mu, con, f)
    g_p = lambda x: grad_P(x, mu, con, con_gr, g)
    
    testp = np.random.randn(5)
    
    #Testing finite differences
    for eps in np.logspace(0, -10, 10):
        print("Eps = {:.2E}".format(eps), (p(x1 + eps*testp) - p(x1))/eps - g_p(x1).dot(testp))
    
    
    while True:
        
        print('Iteration {}'.format(k))
        x1 = sm.bfgs(p, g_p, x0, TOL = 1/k**2)        
#        x1 = sm.steepest_descent(p, g_p, x0, TOL = 1/k**2)
#        x1 = sm.fletcher_reeves(p, g_p, x0, TOL = 1/k**2)
        
        # lagrange multipliers
        lagrange = mu/con(x1)
        
        # convergence test
        # test for KKT
        if check_KKT(x1, lagrange, l_eigen, cf, g):
            return x1
        
        # new penalty parameter
        mu = 1/2*mu
        
        k += 1

        if k > 10000:
            print('No convergence in {} steps'.format(k))
            break
        
        x0 = x1
        
