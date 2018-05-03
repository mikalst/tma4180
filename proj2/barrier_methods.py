#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import search_methods as sm

"""
Implementation of:
    unconstrained primal barrier method
"""

def P(x, mu, con, f):
    """Log-barrier function
        
    Parameters
    ----------
    x : variable.
    mu : penalty parameter.
    con : constraint, callable. 
    f : objective function, callable. 
    
    Returns
    ----------
    P(x) : value of log-barrier function."""
    
    if not((con(x) > 0).all()):
        return np.inf
    
    P = f(x) - mu*np.sum(np.log(con(x)), axis = 0)
    return P


def grad_P(x, mu, con, con_gr, g):
    """Log-barrier gradient
    
    Parameters
    ----------
    x : variable.
    mu : penalty parameter.
    con : constraint, callable. 
    con_gr : constraint gradient, callable
    g : objective gradient, callable. 
    
    Returns
    ----------
    grad_P(x) : value of log-barrier function."""

    grad_P = g(x) - mu*np.sum(np.divide(con_gr(x), np.reshape(con(x), newshape=(5, 1))), axis = 0)

    return grad_P


def test_finite_difference_penalty(x1, mu, f, g, cf, cg):
    """Test if the evaluated P(x) is a good match with the gradient grad_P(x)."""
    
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


def check_KKT(x, mult, l, cf, g, TOL):
    """Check for KKT as a stopping criterion with the given tolerance.
        
    Parameters
    ----------
    x : variable.
    mult : lagrange multipliers. 
    l : lambda_low, part of the objective function definition. 
    cf : constraint function, callable
    g : objective gradient, callable. 
    TOL : tolerance for accepting violation in KKT
    
    Returns
    ----------
    kkt : boolean value indicating whether KKT are sufficiently satisfied."""
    
    violation = np.linalg.norm(g(x) - np.array((mult[0] - mult[1] + mult[4]/2*np.sqrt(x[2]/x[1]),
                             mult[2] - mult[3] + mult[4]/2*np.sqrt(x[0]/x[2]), 
                             mult[3]*x[1]/np.sqrt(l**2 + x[1]**2), 0, 0)), 2)
    if violation > TOL:
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
        

def barrier(x0, mu0, cf, cg, l_eigen, f, g, TOL = 1E-5):
    """Implementation of the log-barrier method as described 
        
    Parameters
    ----------
    x0 : initial value of x.
    mu0 : initial penalty parameter
    cf : constraint function, callable.
    cg : constraint gradient, callable.
    l_eigen : lambda_low, a part of the constraint definition
    f : objective function, callable
    g : ojbective gradient, callable
    TOL : tolerance for accepting violation in KKT
    
    Returns
    ----------
    x_k : Point that either satisfies KKT within specified tolerance or required 100 iterations 
          without satsfying KKT within specified tolerance. """
    
    #Set iteration number
    k = 1
    
    #Set a initial penalty parameter 
    mu = mu0
    x1 = x0
    
    #For ease of calling log-barrier function and gradient.
    p =   lambda x: P(x, mu, cf, f)
    g_p = lambda x: grad_P(x, mu, cf, cg, g)
        
    print("Starting barrier...")
    while True:
        
        print('Outer iteration {}'.format(k), end = ", ")
        
        x1, it1, err1 = sm.bfgs(p, g_p, x1, TOL = 1/k**2, max_iter = 200, linesearch_method = "bt")
        
        #Approximate lagrange multipliers
        lagrange = mu/cf(x1)
        
        #KKT convergence test
        if check_KKT(x1, lagrange, l_eigen, cf, g, TOL = TOL):
            print("KKT sufficiently satisfied,", 'x = {}'.format(x1), sep = "\n")
            return x1, k, f(x1)
        
        #Update penalty parameter
        mu = 1/2*mu
        p =   lambda x: P(x, mu, cf, f)
        g_p = lambda x: grad_P(x, mu, cf, cg, g)
        
        k += 1

        #After 100 iterations, the algorithm is considered not to converge
        if k == 101:
            print("Barrier did not converge within {} steps".format(k-1))
            print(x1)
            return x1, k-1, f(x1)
