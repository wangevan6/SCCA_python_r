#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:46:17 2024

@author: water
"""


import numpy as np
import sweep

import numpy as np

def init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, npairs0, alpha_current, beta_current, n, eps=1e-4, d=None):
    p = sigma_X_hat.shape[1]
    q = sigma_Y_hat.shape[1]
    alpha_init = np.zeros((q, 1))
    beta_init = np.zeros((p, 1))
    alpha_current = np.array(alpha_current)
    beta_current = np.array(beta_current)

    npairs0 = alpha_current.shape[1]

    if init_method == "uniform":
        alpha_init = np.ones((q, 1))
        beta_init = np.ones((p, 1))

    if init_method == "random":
        alpha_init = np.random.normal(size=(q * npairs)).reshape(q, 1)
        beta_init = np.random.normal(size=(p * npairs)).reshape(p, 1)

    if init_method == "svd":
        U, s, Vt = np.linalg.svd(sigma_YX_hat, full_matrices=False)
        alpha_init = U[:, npairs0][:, np.newaxis]  # np.newaxis to keep it 2D
        beta_init = Vt.T[:, npairs0][:, np.newaxis]

    if init_method == "sparse":
        # Identify non-zero indices
        id_nz_alpha = np.where(np.sum(np.abs(alpha_current), axis=1) > eps)[0]
        id_nz_beta = np.where(np.sum(np.abs(beta_current), axis=1) > eps)[0]

        # Compute rho.tmp and sigma.YX.tmp
        #print(f'the shape of alpha_current is {alpha_current.shape}')
        #print(f'the shape of sigma_YX_hat is {sigma_YX_hat.shape}')
        #print(f'the shape of beta_current is {beta_current .shape}')
        rho_tmp = alpha_current.T @ sigma_YX_hat @ beta_current
        sigma_YX_tmp = sigma_YX_hat - sigma_Y_hat @ alpha_current @ rho_tmp @ beta_current.T @ sigma_X_hat

        # Default d if missing
        if d is None:
            d = int(np.sqrt(n))

        # Thresholding
        thresh = np.sort(np.abs(sigma_YX_tmp).ravel())[-d]
        row_max = np.max(np.abs(sigma_YX_tmp), axis=1)
        col_max = np.max(np.abs(sigma_YX_tmp), axis=0)

        id_row = np.unique(np.concatenate((id_nz_alpha, np.where(row_max > thresh)[0])))
        id_col = np.unique(np.concatenate((id_nz_beta, np.where(col_max > thresh)[0])))

        # Perform SVD on the submatrix
        sigma_tmp = sigma_YX_tmp[np.ix_(id_row, id_col)]
        U, s, Vt = np.linalg.svd(sigma_tmp, full_matrices=False)

        # Update alpha.init and beta.init
        alpha_init[id_row] = U[:, 0][:, np.newaxis]
        beta_init[id_col] = Vt.T[:, 0][:, np.newaxis]

        # Normalization (Python equivalent of R's sweep operation)
        alpha_scale = np.sqrt((alpha_init.T @ sigma_Y_hat @ alpha_init).item())
        alpha_init /= alpha_scale
        beta_scale = np.sqrt((beta_init.T @ sigma_X_hat @ beta_init).item())
        beta_init /= beta_scale

    return {'alpha_init': alpha_init, 'beta_init': beta_init}

'''

np.random.seed(123)  # For reproducibility, matching R's set.seed(123)

# Create synthetic covariance matrices
sigma_X_hat = np.random.rand(10, 10)
sigma_X_hat = np.dot(sigma_X_hat.T, sigma_X_hat)  # Make it positive semi-definite
sigma_Y_hat = np.random.rand(8, 8)
sigma_Y_hat = np.dot(sigma_Y_hat.T, sigma_Y_hat)  # Make it positive semi-definite
sigma_YX_hat = np.random.rand(8, 10)              # Cross-covariance matrix

# Simulate sparse canonical vectors with a few non-zero entries
alpha_current = np.zeros((8, 2))
beta_current = np.zeros((10, 2))
alpha_current[1, 0] = 1  # Indexing in Python is 0-based
beta_current[2, 0] = 1
alpha_current[4, 1] = -1
beta_current[6, 1] = -1

# Define other parameters for sparse method
init_method = "sparse"
npairs = 5
npairs0 = 2
n = 50
eps = 1e-4
d = 3  # Choosing a small number for the test case

# Assume init1 is the Python function we've written previously that mirrors the R function
init_values = init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, npairs0, alpha_current, beta_current, n, eps, d)

# Print the resulting initial vectors
print("alpha.init:")
print(init_values['alpha_init'])
print("\nbeta.init:")
print(init_values['beta_init'])

'''



