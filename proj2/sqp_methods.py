#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:00:38 2018

@author: mikal
"""

import numpy as np
from scipy.optimize import minimize
import constrained_ellipsoids as ce

CONSTRAINTS = 5


LAMBDA = 1E0
LAMBDB = 1E3

def scipy_constraints_qp(x_k):
    
    lambda_l = LAMBDA
    lambda_h = LAMBDB
    
    constraint1 = {'type': 'ineq',
                   'fun': lambda p: x_k[0] + p[0] - lambda_l}
    
    constraint2 = {'type': 'ineq',
                   'fun': lambda p: lambda_h - (x_k[0] + p[0])}
    
    constraint3 = {'type': 'ineq',
                   'fun': lambda p: (x_k[2]+p[2]) - lambda_l}
    
    constraint4 = {'type': 'ineq',
                   'fun': lambda p: lambda_h - (x_k[2] + p[2])}
    
    constraint5 = {'type': 'ineq',
                   'fun': lambda p: np.power((x_k[0]+p[0])*(x_k[2]+p[2]), 0.5) - np.power(lambda_l**2 + (x_k[1]+p[1])**2, 0.5)}       

    return [constraint1, constraint2, constraint3, constraint4, constraint5]


def scipy_qp_solver(g, x_k, B_k):
    
    f = lambda p: 1/2*np.dot(p, np.dot(B_k, p)) - g(x_k).dot(p)
    p_0 = np.zeros(5)
    
    constraints = scipy_constraints_qp(x_k)
    
    p = minimize(f, p_0, method = 'COBYLA', constraints = constraints).x
    
    print("p = ", p)
    
    return x
    


def linesearch_sqp(x_0, l_0, f, g, cf, cg):
    
    def lagrangian_x(x, l):
        return g(x) + np.dot(l, cg(x, LAMBDA))
    
    eta = 0.4
    tau = 0.5
    
#    f_k = f(x_0)
#    g_k = g(x_0)
#    c_k = cf(x_0)
#    A_k = cg(x_0)
    
    B_k = np.identity(len(x_0))
    
    #Functions and values that I need to find
    mu_k = 1
    phi_1 = lambda x, mu: f(x) + mu * np.linalg.norm(cf(x, LAMBDA, LAMBDB), 1)
    D_1 = lambda phi, mu, p: g(x).dot(p) - mu * np.linalg.norm(cf(x, LAMBDA, LAMBDB), 1)

    #End of things I need to find
    
    #Find some hessian approximation using (18.13)
    
    x_k = x_0
    l_k = l_0
    
    while np.linalg.norm(g(x_k), 2) > 1E-3:
#        p_k = 
        p_k = scipy_qp_solver(g, x_k, B_k)
        l_hat = np.linalg.lstsq(cg(x_k, LAMBDA), g(x_k))[0]
        print(l_hat)
        p_l = l_hat - l_k
        #Choose mu_k to satisfy (18.36) with sigma = 1
        alpha_k = 1
        while phi_1(x_k + alpha_k*p_k, mu_k) > phi_1(x_k, mu_k) + eta*alpha_k*D_1(phi_1(x_k, mu_k), mu_k, p_k):
            alpha_k = tau * alpha_k
        x_k_old = x_k
        x_k = x_k_old + alpha_k * p_k
        l_k = l_k + alpha_k * p_l
        
#        f_k = f(x_k)
#        g_k = g(x_k)
#        c_k = cf(x_k)
#        A_k = cg(x_k)
        
        s_k = alpha_k * p_k
        
        print("grad = {}".format(g(x_k)), "impr = {}".format(s_k), sep="\n")
        
        y_k = lagrangian_x(x_k, l_k) - lagrangian_x(x_k_old, l_k)
        B_k = B_k - np.outer(B_k.dot(s_k), s_k).dot(B_k)/(s_k.T.dot(B_k).dot(s_k)) + np.outer(y_k, y_k)/np.dot(y_k, y_k)
        
        print('x = ', x)
    
    return x_k

if __name__ == "__main__":
    nz = 500
    N = 2
    z, w = ce.generate_points(nz, N)
    mu0 = 2
    
    A = np.array([[1, -10], [-10, 1]])
    c = np.array([.0, .0])

    
    lambda_l = 1E0
    lambda_h = 1E3
    
    #Initialize with feasible initial point
    x = np.zeros(int(N*(N+1)/2 + N))
    x[0] = (lambda_l + lambda_h)/2
    x[1] = 0
    x[2] = (lambda_l + lambda_h)/2
    
    
    f, g = ce.setmodelzw(z, w, x)
        
    x = linesearch_sqp(x, np.zeros(5), f, g, ce.cf, ce.cg)
    
#    x2 = minimize(f, x, jac=g, method = 'SLSQP', constraints = scipy_constraints(lambda_l, lambda_h))
#    x3_unconstrained, it3, err3 = sm.bfgs(f, g, x)
