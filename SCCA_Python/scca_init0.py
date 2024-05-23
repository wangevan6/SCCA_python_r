#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:08:12 2024

@author: water
"""


import numpy as np
import sweep

np.random.seed(42)

def init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, n, d=None):
    
    '''
     The function init0 finds the initial value when no canonical pairs have been obtained. If init.method="sparse", 
     only one pair of initial value will be returned. For other options of init.method, the number of pairs of initial
     values can be specified with the argument npairs.
     
     sigma.YX.hat: Estimated cross-covariance matrix between Y and X.
     sigma.X.hat: Estimated covariance matrix of X.
     sigma.Y.hat: Estimated covariance matrix of Y.
     init.method: A string indicating which initialization method to use.
     npairs: Number of initial pairs needed.
     n: Sample size (used in sparse initialization).
     d: A threshold parameter used in sparse initialization (optional).
     
    '''
    q, p = sigma_Y_hat.shape[1], sigma_X_hat.shape[1]

    if init_method == 'svd':
        u, _, v = np.linalg.svd(sigma_YX_hat, full_matrices=False)
        alpha_init = u[:, :npairs]
        beta_init = v.T[:, :npairs]
    
    
    if init_method == 'uniform':
        alpha_init = np.ones((q, npairs))
        beta_init = np.ones((p, npairs))
    
    if init_method == 'random':
        alpha_init = np.random.normal(size=(q, npairs))
        beta_init = np.random.normal(size=(p, npairs))

    if init_method == 'sparse':
        if d is None:
            d = int(np.sqrt(n))
        thresh = np.sort(np.abs(sigma_YX_hat).ravel())[::-1][d - 1]
        row_max = np.max(np.abs(sigma_YX_hat), axis=1)
        col_max = np.max(np.abs(sigma_YX_hat), axis=0)
        
        selected_rows = np.where(row_max > thresh)[0]
        selected_cols = np.where(col_max > thresh)[0]
        reduced_sigma_YX_hat = sigma_YX_hat[np.ix_(selected_rows, selected_cols)]
        
        u, _, v = np.linalg.svd(reduced_sigma_YX_hat, full_matrices=False)
        alpha1_init = np.zeros(q)
        beta1_init = np.zeros(p)
        alpha1_init[selected_rows] = u[:, 0]
        beta1_init[selected_cols] = v.T[:, 0]

        alpha_init = np.zeros((q, npairs))
        beta_init = np.zeros((p, npairs))
        alpha_init[:, 0] = alpha1_init
        beta_init[:, 0] = beta1_init

    # Scaling alpha_init and beta_init
    alpha_scale = np.diag(alpha_init.T @ sigma_Y_hat @ alpha_init)
    alpha_init = sweep.sweep(alpha_init, margin=1, stats=np.sqrt(alpha_scale), operation='/')
    #alpha_init = alpha_init / np.sqrt(alpha_scale)[:, np.newaxis]
    beta_scale = np.diag(beta_init.T @ sigma_X_hat @ beta_init)
    beta_init = sweep.sweep(beta_init, margin=1, stats=np.sqrt(beta_scale), operation='/')
    #beta_init = beta_init / np.sqrt(beta_scale)[:, np.newaxis]
    return {'alpha_init': alpha_init, 'beta_init': beta_init}


