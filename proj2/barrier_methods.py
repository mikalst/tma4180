#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import search_methods as sm


# every constraint is a inequality constraint, dont bother to check

# evaluate the modified objective function
def P(x, mu, con, f):
    if not((con(x) > 0).all()):
        return np.inf
    
    P = f(x) - mu*np.sum(np.log(con(x)), axis = 0)
    return P

# gradient of modidied objective function
def grad_P(x, mu, con, con_gr, g):
    grad_P = g(x) - mu*np.sum(np.divide(con_gr(x), np.reshape(con(x), newshape=(5, 1))), axis = 0)

    return grad_P


# check for KKT as a stopping criterion
def check_KKT(x, mult, l, cf, g):
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
    k = 1
    # set a initial penalty parameter 
    mu = mu0
    x1 = x0
        
    p =   lambda x: P(x, mu, cf, f)
    g_p = lambda x: grad_P(x, mu, cf, cg, g)
    

    #Testing finite differences
#    testp = np.random.randn(5)
#    for eps in np.logspace(0, -10, 11):
#        print("Eps = {:.2E}".format(eps), (p(x1 + eps*testp) - p(x1))/eps - g_p(x1).dot(testp))    
    
    
    np.seterr('ignore')
    
    while True:
        
        print('Iteration {}'.format(k))
        
        x1, it1, err1 = sm.bfgs(p, g_p, x0, TOL = 1/k**2)    
#        x1 = sm.steepest_descent(p, g_p, x0, TOL = 1/k**2)
#        x1 = sm.fletcher_reeves(p, g_p, x0, TOL = 1/k**2)
        
        # approximate lagrange multipliers
        lagrange = mu/cf(x1)
        
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
        
