#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:00:38 2018

@author: mikal
"""

import numpy as np
from scipy.optimize import minimize


def equality_constrained_qp(g, x_k, B_k, cf, cg, W_k):
    W_full = set([0, 1, 2, 3, 4])
    if len(W_k) == 0: #Unconstrained
        return np.linalg.solve(B_k, c)
    
    for i in W_full - W_k:
        
    
    block = np.block([G, -cf.transpose


def active_set_method_convex_qp(g, x_k, B_k, cf, cg):
    qf = lambda x: np.dot(np.dot(x, B_k), x) + g.dot(x)
    qg = lambda x: 2*np.dot(B_k, x) + g
    
    W_0 = set([])
    
    while np.linalg.norm(qg, 2) > 1E-8:
        
    
    lambda_l = 1E-3
    lambda_h = 1E3
    
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
        
    x0 = minimize(q, x_k, constraints = [constraint1,
                                         constraint2,
                                         constraint3,
                                         constraint4,
                                         constraint5])

    x = x0.x
    
    
    return x0

def linesearch_sqp(x_0, l_0, f, g, cf, cg):
    
    def lagrangian_x(x, l):
        return g(x) + np.dot(l, cg(x))
    
    eta = 0.4
    tau = 0.5
    
#    f_k = f(x_0)
#    g_k = g(x_0)
#    c_k = cf(x_0)
#    A_k = cg(x_0)
    
    B_k = np.identity(len(x_0))
    
    #Functions and values that I need to find
    mu_k = 1
    phi_1 = lambda x, mu: f(x) + mu * np.linalg.norm(cf(x), 1)
    D_1 = lambda x, mu, p: g(x).dot(p) - mu * np.linalg.norm(cf(x), 1)

    #End of things I need to find
    
    #Find some hessian approximation using (18.13)
    
    x_k = x_0
    l_k = l_0
    
    while np.linalg.norm(g(x_k), 2) > 1E-3:
        p_k, l_hat = active_set_method_convex_qp(g, x_k, B_k, cf, cg)
        p_l = l_hat - l_k
        #Choose mu_k to satisfy (18.36) with sigma = 1
        alpha_k = 1
        while phi_1(x_k + alpha_k*p_k, mu_k) > phi_1(x_k, mu_k) + eta*alpha_k*D_1(phi_1(x_k, mu_k)*p_k):
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
        B_k = B_k - B_k.dot(s_k).outer(s_k.T).dot(B_k)/(s_k.T.dot(B_k).dot(s_k)) + np.outer(y_k, y_k)/np.dot(y_k, y_k)
    
    return x_k
        