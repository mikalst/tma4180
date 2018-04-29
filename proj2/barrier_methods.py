#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import search_methods as sm



def P(x, mu, con, f):
    """Log-barrier function"""
    if not((con(x) > 0).all()):
        return np.inf
    
    P = f(x) - mu*np.sum(np.log(con(x)), axis = 0)
    return P


def grad_P(x, mu, con, con_gr, g):
    """Log-barrier gradient"""
    grad_P = g(x) - mu*np.sum(np.divide(con_gr(x), np.reshape(con(x), newshape=(5, 1))), axis = 0)

    return grad_P


def test_finite_difference_penalty(x1, mu, f, g, cf, cg):
    np.random.seed()
    testp = 0.01*np.random.randn(5)
        
    p =   lambda x: P(x, mu, cf, f)
    g_p = lambda x: grad_P(x, mu, cf, cg, g)
    
    print("\nTesting f and g by fin. differences...")
    maxval = -np.inf
    for eps in np.logspace(-2, -9, 8):
        err = (p(x1 + eps*testp) - p(x1))/(eps) - g_p(x1).dot(testp)
        maxval = max(err, maxval)
        print("Eps = {:.2E}".format(eps), '{:.5E}'.format(err))
    if (maxval <= 1E-1):
        print("===> F and G match\n")


def check_KKT(x, mult, l, cf, g):
    """Check for KKT as a stopping criterion"""

    # set a tolerance for checking KKT 
    TOL = 10E-3
    
    if np.linalg.norm(g(x) - np.array((mult[0] - mult[1] + mult[4]/2*np.sqrt(x[2]/x[1]),
                             mult[2] - mult[3] + mult[4]/2*np.sqrt(x[0]/x[2]), 
                             mult[3]*x[1]/np.sqrt(l**2 + x[1]**2), 0, 0)), 2) > TOL:
        return False
    con = cf(x)
    if (con < 0).any():
        return False
    if (mult < 0).any():
        return False
    for i in range(len(mult)):
        if mult[i]*con[i] > TOL:
            return False
    return True
        

def barrier(x0, mu0, cf, cg, l_eigen, f, g):
    
    # Set iteration number
    k = 1
    
    # Set a initial penalty parameter 
    mu = mu0
    x1 = x0
    
    # For ease of calling log-barrier function and gradient.
    p =   lambda x: P(x, mu, cf, f)
    g_p = lambda x: grad_P(x, mu, cf, cg, g)
    

    # Testing finite differences.
    test_finite_difference_penalty(x1, mu, f, g, cf, cg) 
    
    
    np.seterr('ignore')
    
    print("Starting barrier...")
    while True:
        
        print('Outer iteration {}'.format(k), end = ", ")
        
        x1, it1, err1 = sm.bfgs(p, g_p, x0, TOL = 1/k**2, max_iter = 100)    
#        x1, it1, err1 = sm.steepest_descent(p, g_p, x0, TOL = 1/k**2)
#        x1, it1, err1 = sm.fletcher_reeves(p, g_p, x0, TOL = 1/k**2)
        
        # approximate lagrange multipliers
        lagrange = mu/cf(x1)
        
        # KKT convergence test
        if check_KKT(x1, lagrange, l_eigen, cf, g):
            print("KKT sufficiently satisfied,", 'x = {}'.format(x1), sep = "\n")
            return x1
        
        # update penalty parameter
        mu = 1/2*mu
        
        k += 1

        if k > 30:
            print('No convergence in {} steps'.format(k))
            break
        
        x0 = x1
