#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:53:48 2018

@author: Lars
"""

def P(x, mu, con, z, w, f):
    P = f(z, w, x)
    for i in range(len(con)):
        if con[i]['type'] == 'ineq':
            P += mu*con[i]['fun']
    return P

def grad_P(x, mu, con, z, w):
    grad_P = jacobi(z, w, x)
    for i in range(len(con)):
        if con[i]['type'] == 'ineq':
            grad_P -= mu*con[i]['jac']
    return grad_P
     
def barrier(x0, mu0, f):
    k = 1
    x1 = x0 
    mu = mu0
    
    lambda p: P(x, mu, con, z, w, f)
    lambda g_p: grad_P(x, mu, con, z, w)
    
    while True:
        x = bfgs(p, g_p, TOL = 1/k) # have to modify bfgs to yield feasable points
        
        # lagrange multipliers
        z = 