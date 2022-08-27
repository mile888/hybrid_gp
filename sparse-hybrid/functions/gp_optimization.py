#!/usr/bin/env python

####  GPR optimization  ####


import time
import numpy as np
import random

import scipy
from scipy.optimize import minimize
from scipy import spatial as spt


def cov_kernel(X, Xm, ell, sf2):
    """ Squared Exponential Automatic Relevance Determination (SE ARD) covariance kernel K.

    # Arguments:
        X:    Inputs training data matrix of size (N x Nx) - Nx is the number of inputs to the GP.
        ell:  Length scales vector of size Nx.
        sf2:  Signal variance (scalar)
    """
    dist = 0
    n, D = X.shape
    m, _ = Xm.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        xm = Xm[:, i].reshape(m, 1)
        dist = (spt.distance.cdist(x, xm, metric='sqeuclidean') / ell[i]**2) + dist
    return sf2 * np.exp(-.5 * dist)



def NegLowerBound(hyper, a):
    ''' Find negative lower bound (negative so we can use
        minimisation optimisation functions)
    '''

    X, Y, Xs, N, M = a 
    n, D = X.shape
    m, _ = Xs.shape
    
    Y = Y.reshape(-1,1)
    
    ell = hyper[:D]
    sf2 = hyper[D]**2
    sn2 = hyper[D + 1]**2
    
    K_MM = cov_kernel(Xs, Xs, ell, sf2)     
    InvK_MM = np.linalg.inv(K_MM)                 
    K_NM = cov_kernel(X, Xs, ell, sf2)  
    K_MN = np.transpose(K_NM)                     

    A = np.dot(K_NM, np.dot(InvK_MM, K_MN))

    B = np.zeros(n)
    for i in range(n):
        B[i] = 1 - A[i, i]

    C = A+np.eye(n)*sn2
    Sign, LogDetC = np.linalg.slogdet(C)
    LogDetC = Sign*LogDetC
    
    NLB = -(-0.5*LogDetC - 0.5*np.dot(Y.T, np.dot(np.linalg.inv(C), Y)) - 1/(2*sn2)*np.sum(B))
    
    return NLB.ravel()


def train_gp(X, Y, M, NoCandidates, multistart=1, hyper_init=None, optimizer_opts=None):
    """ Train hyperparameters

    Maximum likelihood estimation is used to optimize the hyperparameters of
    the Gaussian Process. Sequential Least SQuares Programming (SLSQP) is used
    to find the optimal solution.

    A uniform prior of the hyperparameters are assumed and implemented as
    limits in the optimization problem.

    NOTE: This version only support a zero-mean function.

    # Arguments:
        X: Training data matrix with inputs of size (N x Nx),
            where Nx is the number of inputs to the GP.
        Y: Training data matrix with outputs of size (N x Ny),
            with Ny number of outputs.

    # Return:
        opt: Dictionary with the optimal hyperparameters [ell_1 .. ell_Nx sf sn].
    """
    
    N, Nx = X.shape
    Ny = Y.shape[1]
    
    h_ell   = Nx    
    h_sf    = 1     
    h_sn    = 1    
    num_hyp = h_ell + h_sf + h_sn 

    options = {'disp': True, 'maxiter': 10000}
    if optimizer_opts is not None:
        options.update(optimizer_opts)

    hyp_opt = np.zeros((Ny, num_hyp))
    invK = np.zeros((Ny, M, M))
    alpha = np.zeros((Ny, M))
    chol = np.zeros((Ny, M, M))
    Xs_all = np.zeros((Ny, M, Nx))
    Ys_all = np.zeros((Ny, M))
    mu_m_all = np.zeros((Ny, M))
    A_m_all = np.zeros((Ny, M, M))

    print('\n________________________________________')
    print('# Optimizing hyperparameters (N=%d)' % N )
    print('----------------------------------------')
    for output in range(Ny):
        meanF     = np.mean(Y[:, output])
        lb        = -np.inf * np.ones(num_hyp)
        ub        = np.inf * np.ones(num_hyp)
        lb[:Nx]    = 1e-2
        ub[:Nx]    = 2e2
        lb[Nx]     = 1e-8
        ub[Nx]     = 1e2
        lb[Nx + 1] = 1e-10
        ub[Nx + 1] = 1e-2
        bounds = np.hstack((lb.reshape(num_hyp, 1), ub.reshape(num_hyp, 1)))

        if hyper_init is None:
            hyp_init = np.zeros((num_hyp))
            hyp_init[:Nx] = np.std(X, 0)
            hyp_init[Nx] = np.std(Y[:, output])
            hyp_init[Nx + 1] = 1e-5
        else:
            hyp_init = hyper_init[output, :]
            
            
       
        for r in range(NoCandidates):
            random.seed(r)
            Y_tr = Y[:, output]
            indices = random.sample(range(N), M)
            X_candidate = X[indices]
            
            a = (X, Y_tr, X_candidate, N, M)
            LB = -NegLowerBound(hyp_init, a)
            

            if r == 0:              
                Xs = X_candidate    
                Ys = Y_tr[indices]     
                LB_best = LB        
            else:
                if LB > LB_best:      
                    Xs = X_candidate  
                    Ys = Y_tr[indices]  
                    LB_best = LB
            #print(Xs)

        
        a = (X, Y[:, output], Xs, N, M)  
        Xs_all[output, :] = Xs
        Ys_all[output, :] = Ys

        obj = np.zeros((multistart, 1))
        hyp_opt_loc = np.zeros((multistart, num_hyp))
        for i in range(multistart):
            solve_time = -time.time()
            res = minimize(NegLowerBound, x0=hyp_init, args=(a, ),
                           method='SLSQP', options=options, bounds=bounds, tol=1e-15)
            obj[i] = res.fun
            hyp_opt_loc[i, :] = res.x
        solve_time += time.time()
        print("* Output %d:  %f s" % (output+1, solve_time))

        hyp_opt[output, :]   = hyp_opt_loc[np.argmin(obj)]
        ell = hyp_opt[output, :Nx]
        sf2 = hyp_opt[output, Nx]**2
        sn2 = hyp_opt[output, Nx + 1]**2

       
        K = cov_kernel(Xs, Xs, ell, sf2)
        K = K + sn2 * np.eye(M)
        K = (K + K.T) * 0.5   
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            print("K matrix is not positive definit, adding jitter!")
            K = K + np.eye(M) * 1e-8
            L = np.linalg.cholesky(K)
        invL = np.linalg.solve(L, np.eye(M))
        invK[output, :, :] = np.linalg.solve(L.T, invL)
        chol[output] = L
        alpha[output] = np.linalg.solve(L.T, np.linalg.solve(L, Ys))
        mu_m, A_m = phi_opt(ell, sf2, sn2, Xs, X, Y[:, output])
        
        A_m_all[output, :, :] = A_m
        mu_m_all[output, :] = mu_m
        
        
        
    print('----------------------------------------')

    opt = {}
    opt['hyper'] = hyp_opt
    opt['invK'] = invK
    opt['alpha'] = alpha
    opt['chol'] = chol
    opt['Xs'] = Xs_all
    opt['Ys'] = Ys_all.T
    opt['mu_m_all'] = mu_m_all.T
    opt['A_m_all'] = A_m_all
    return opt


def jitter(d, value=1e-6):
    return jnp.eye(d) * value

def phi_opt(ell, sf2, sn2, X_m, X, y):
    """Optimize mu_m and A_m """
    precision = (1.0 / sn2 ** 2)

    K_mm = cov_kernel(X_m, X_m, ell, sf2) 
    K_mm_inv = np.linalg.inv(K_mm)
    K_nm = cov_kernel(X, X_m, ell, sf2)
    K_mn = K_nm.T
    
    Sigma = np.linalg.inv(K_mm + precision * K_mn @ K_nm)
    
    mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
    A_m = K_mm @ Sigma @ K_mm    
    
    return mu_m, A_m

