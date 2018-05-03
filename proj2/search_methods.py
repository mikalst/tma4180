#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of:
    bactracking linesearch
    bisection linesearch
    zoom
    steepest descent
    bfgs
    fletcher-reeves
"""

import numpy as np


def backtracking_linesearch(f, g, x_k, p_k, g_k):
    """Backtracking linesearch
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    p_k : direction of search.
    g_k : gradient evaluated at inital point.
    
    Returns
    ----------
    x_k+1 : point satisfying sufficient decrease, or point at which the linesearch 
            was terminated
    success : Boolean value indicating the convergence of the linesearch."""
    
    f0 = f(x_k)
    alpha = 1
    c1 = 0.2

    sd = False
    while(not sd):
        alpha *= 0.5
        sd = f(x_k + alpha * p_k) <= f0 + c1 * alpha * g_k.T@p_k
    
    #If increment becomes negligible, i.e. where the computation
    # (x_k == x_k + alpha * p_k) evaluates to True. This effectively means that
    # we are not making progress and the direction for search should reset to 
    # steepest descent. 
    if (np.max(np.abs(alpha * p_k / x_k)) < 2.2E-16):
        return x_k + alpha*p_k, False
    
    return x_k + alpha*p_k, True


def zoom(f, g, x_k, p_k, alpha_lo, alpha_hi, c1, c2):
    """Zoom
    
        Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    p_k : direction of search.
    alpha_lo: lower value of alpha.
    alpha_hi: upper value of alpha.
    c1 : 1st parameter of the wolfe conditions.
    c2 : 2nd parameter of wolfe conditions. 
    
    Returns
    ----------
    x_k+1 : point satisfying strong wolfe, or point at which the linesearch 
            was terminated
    success : Boolean value indicating the convergence of the linesearch."""
    
    f0 = f(x_k)
    g0 = g(x_k)
        
    while True:
        
        alpha_j = (alpha_lo + alpha_hi)/2

        if (np.abs(alpha_lo - alpha_hi) == 0.):
            return x_k + alpha_j * p_k, False
        
        f_j = f(x_k + alpha_j*p_k)
        
        # Check if not sufficient decrease
        if (f_j > f0 + c1*alpha_j*g0.dot(p_k)) or f_j >= f(x_k + alpha_lo * p_k):
            alpha_hi = alpha_j

        else:
            g_j = g(x_k + alpha_j * p_k)
            
            # Check curvature
            if np.abs(g_j.dot(p_k)) <= -c2*g0.dot(p_k):
                return x_k + alpha_j * p_k, True
            
            if g_j.dot(p_k)*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                
            alpha_lo = alpha_j


def linesearch(f, g, x_k, p_k, c1, c2, wolfe='s'):
    """Linesearch
    
        Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    p_k : direction of search.
    c1 : 1st parameter of the wolfe conditions.
    c2 : 2nd parameter of wolfe conditions. 
    wolfe : specification of wolfe conditions, {s, w}
    
    Returns
    ----------
    x_k+1 : point satisfying strong wolfe, or point at which the linesearch 
            was terminated
    success : Boolean value indicating the convergence of the linesearch."""

    alpha_0 = 0
    alpha_max = np.inf
    alpha_i = 1
    
    f0 = f(x_k)
    g0 = g(x_k)
    
    f_last = f0
    alpha_last = alpha_0
    i = 1
    
    #Check if p_k is a valid direction. 
    if not(g(x_k).dot(p_k) < 0):
        return x_k, False
    
    while True:
        
        if (alpha_i < 1E-16):
            print("very small step")
        
        f_i = f(x_k + alpha_i * p_k)
        if f_i > f0 + c1*alpha_i*g0.dot(p_k) or (f_i >= f_last and i > 1):
            return zoom(f, g, x_k, p_k, alpha_last, alpha_i, c1, c2)
        g_i = g(x_k + alpha_i * p_k)
        if np.abs(g_i.dot(p_k)) <= -c2*g0.dot(p_k):
            return x_k + alpha_i * p_k, True
        if g_i.dot(p_k) >= 0:
            if wolfe=='s':
                return zoom(f, g, x_k, p_k, alpha_i, alpha_last, c1, c2)
            elif wolfe=='w':
                return x_k + alpha_i * p_k, True
            else:
                print('Wolfe condition should be {\'s\', \'w\'}, \ndefaulting to strong ...')
                return zoom(f, g, x_k, p_k, alpha_i, alpha_last, c1, c2)
        
        alpha_last = alpha_i
        if alpha_max == np.inf:
            alpha_i *= 2
        else:
            alpha_i = (alpha_i + alpha_max)/2
        i += 1


def steepest_descent(f, g, x, TOL = 1e-3, max_iter = 9999):
    """Steepest descent algorithm for unconstrained minimization.
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    TOL : maximum Euclidian-norm of gradient at local minimum.
    max_iter : maximum number of iterations.
    
    Returns
    ----------
    x* : obtained minimum.
    iterations : iterations used.
    f(x_k*) : value of objective at local minimum."""
    
    print("SD \nf(x_0) = ", f(x))

    g_k = g(x)
    if np.linalg.norm(g_k) < TOL:
        return x
    
    x_k = x
    p_k = -g_k

    iterations = 1

    while True:
        x_k, success = backtracking_linesearch(f, g, x_k, p_k, g_k)
        g_k = g(x_k)
        
#        print(g_k)
        
        iterations += 1
        if np.linalg.norm(g_k) < TOL or iterations >= max_iter:
            break
        
        p_k = -g_k
    
    print("f(x_{}) = {}".format(iterations, f(x_k)))
    
    return x_k, iterations, f(x_k)


def bfgs(f, g, x, TOL = 1e-3, max_iter = 9999, linesearch_method = "ww"):
    """BFGS algorithm for unconstrained minimization.
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x : initial point.
    TOL : maximum Euclidian-norm of gradient at local minimum.
    max_iter : maximum number of iterations.
    linesearch_method : method to be used in linesearch. {bt, ww}
    
    Returns
    ----------
    x* : obtained minimum.
    iterations : iterations used.
    f(x_k*) : value of objective at local minimum."""
    
    #print("BFGS \nf(x_0) = ", f(x))
    
    I = np.identity(len(x))
    iterations = 1
    reset_to_SD_counter = 0
    x_k = x
    g_k = g(x)
    H_k =  I

    while True:
        x0 = x_k
        g0 = g_k    
        p_k = - H_k@g_k
        
        if linesearch_method == "bt":
            x_k, ls_success = backtracking_linesearch(f, g, x_k, p_k, g_k)
            
        elif linesearch_method == "ww":
            x_k, ls_success = linesearch(f, g, x_k, p_k, 1E-4, 0.9, wolfe='w')
        
        g_k = g(x_k)
        
        iterations += 1
        if np.linalg.norm(g_k) < TOL or iterations >= max_iter:
            break
        
        #Reset to steepest descent either if our chosen direction is not a direction
        if not(ls_success): #(g_k.dot(p_k) / (np.linalg.norm(g_k) * np.linalg.norm(p_k))) < 1e-10
            reset_to_SD_counter += 1
            H_k = I
            continue
 
        y = g_k - g0; 
        s = x_k - x0;
        
        if (y.dot(s) <= 0):
            H_k = H_k
        else:
            rho = 1/y.dot(s)
            H_k = (I - rho*np.outer(s, y))@H_k@(I - rho*np.outer(y, s)) + rho*np.outer(s, s)
        
#    if np.linalg.norm(g(x_k)) == np.nan:
#        return steepest_descent(f, g, x, TOL, max_iter)
    
    #print("f(x_{}) = {}".format(iterations, f(x_k)))
    print('BFGS iterations: {}'.format(iterations), 'reset SD: {}'.format(reset_to_SD_counter))
    return x_k, iterations, f(x_k)


def fletcher_reeves(f, g, x, TOL = 1e-3, max_iter = 9999):
    """Fletcher-Reeves algorithm for unconstrained minimization.
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x : initial point.
    TOL : maximum Euclidian-norm of gradient at local minimum.
    max_iter : maximum number of iterations.
    
    Returns
    ----------
    x* : obtained minimum.
    iterations : iterations used.
    f(x_k*) : value of objective at local minimum."""
    
    print("FR \nf(x_0) = ", f(x))
    x0 = x
    g0 = g(x)
          
    p0 = -g0
    iterations = 1
      
    while True: 
    
        x_next, success = linesearch(f, g, x0, p0, 1E-4, 0.45, wolfe='s')
        g_next = g(x_next)
          
        if np.linalg.norm(g_next) < TOL or iterations >= max_iter:
            break
          
        beta_fr = g_next.dot(g_next)/(g0.dot(g0))
        beta_pr = g_next.dot(g_next - g0) / (np.linalg.norm(g0)**2)
          
        if beta_pr < -beta_fr:
            beta = -beta_fr
        elif np.abs(beta_pr) <= beta_fr:
            beta = beta_pr
        else:
            beta = beta_fr

        p_next = -g_next + beta*p0
        
        p0 = p_next
        x0 = x_next
        g0 = g_next
          
        iterations += 1
    
    print("f(x_{}) = {}".format(iterations, f(x_next)))
    
    return x_next, iterations, f(x_next)