#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:25:27 2024

@author: oujakusui
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from scipy.linalg import svd
from scipy.stats import zscore
from scca_findomega import find_Omega
from scca_init0 import init0
from scca_init1 import init1
from SCCA_scca_solution import SCCA_solution

def SCCA(x, y, lambda_alpha, lambda_beta, alpha_init=None, beta_init=None, niter=100, npairs=1, init_method="sparse", alpha_current=None, beta_current=None, standardize=True, eps=1e-4):

    p = x.shape[1]
    q = y.shape[1]
    n = x.shape[0]
    #print(f'the value of p  is {p}')
    x = scale(x, with_mean=True, with_std=standardize)
    y = scale(y, with_mean=True, with_std=standardize)

    #sigma_YX_hat = np.cov(y, x, rowvar=False)
    sigma_YX_hat =  np.cov(y.T, x.T)[:q, q:] # The reason behind this is unknown
    sigma_X_hat = np.cov(x, rowvar=False)
    sigma_Y_hat = np.cov(y, rowvar=False)
    #print(f'the shape of sigma_YX_hat is {sigma_YX_hat.shape}')
    alpha = np.zeros((q, npairs))
    beta = np.zeros((p, npairs))
    rho = np.zeros((npairs, npairs))

    if isinstance(init_method, list):
        init_method = init_method[0]

    if alpha_current is not None:
        npairs0 = alpha_current.shape[1]
        alpha[:, :npairs0] = alpha_current
        beta[:, :npairs0] = beta_current
    else:
        npairs0 = 0

    if alpha_init is None:
        if alpha_current is None:
            # Placeholder for init0 function
            obj_init = init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method=init_method, npairs=npairs, n=n)
            alpha_init = obj_init['alpha_init']
            beta_init = obj_init['beta_init']
        else:
            alpha_current = np.asmatrix(alpha_current)
            beta_current = np.asmatrix(beta_current)
            # Placeholder for init1 function
            obj_init = init1(sigma_YX_hat=sigma_YX_hat, sigma_X_hat=sigma_X_hat, sigma_Y_hat=sigma_Y_hat,
                             init_method=init_method, npairs=npairs, npairs0=npairs0, alpha_current=alpha_current,
                             beta_current=beta_current, n=n, eps=eps)
            alpha_init = obj_init['alpha_init']
            beta_init = obj_init['beta_init']
            alpha_current = np.array(alpha_current)
            beta_current = np.array(beta_current)


    n_iter_converge = np.zeros(npairs - npairs0)
###############################################################################
    for ipairs in range(npairs0, npairs):
        #print(f"Processing pair {ipairs+1} of {npairs}")  
        alpha_init = np.array(alpha_init)
        beta_init = np.array(beta_init)

        # Placeholder for find_Omega function
        omega = find_Omega(sigma_YX_hat, ipairs, alpha=alpha[:, :ipairs], beta=beta[:, :ipairs], y=y, x=x)

        x_tmp = omega.dot(x)
        y_tmp = omega.T.dot(y)
        #print(f"the value of lambda_alpha is {lambda_alpha}")
        #lambda_alpha0 = lambda_alpha[ipairs - npairs0]
        #lambda_beta0 = lambda_beta[ipairs - npairs0]
        try:
            lambda_alpha0 = lambda_alpha[ipairs - npairs0]
        except IndexError:  # Caught if lambda_alpha is not subscriptable
            lambda_alpha0 = lambda_alpha
        try:
            lambda_beta0 = lambda_alpha[ipairs - npairs0]
        except IndexError:  # Caught if lambda_alpha is not subscriptable
            lambda_beta0 = lambda_alpha
###############################################################################
        alpha0 = alpha_init
        beta0 = beta_init

        # Placeholder for SCCA_solution function
        obj = SCCA_solution(x=x, y=y, x_Omega=x_tmp, y_Omega=y_tmp, alpha0=alpha0, beta0=beta0, lambda_alpha=lambda_alpha0,
                            lambda_beta=lambda_beta0, niter=niter,eps=eps)

        alpha[:, ipairs] = obj['alpha'].flatten()
        beta[:, ipairs] = obj['beta'].flatten()
        n_iter_converge[ipairs - npairs0] = obj['niter']

        if ipairs < npairs and init_method == "sparse":
            # Placeholder for init1 function call
            #print(f'the shape of sigma_YX_hat is {sigma_YX_hat.shape}')
            obj_init = init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method=init_method,
                             npairs=npairs, npairs0=ipairs, alpha_current=alpha[:, :ipairs], beta_current=beta[:, :ipairs],n=n )
            alpha_init = obj_init['alpha_init']
            beta_init = obj_init['beta_init']

    return {
        'alpha': alpha,
        'beta': beta,
        'alpha_init': alpha_init,
        'beta_init': beta_init,
        'n_iter_converge': n_iter_converge
    }
