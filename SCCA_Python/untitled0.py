#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:28:11 2024

@author: oujakusui
"""

import numpy as np

# Given settings
p = 300
q = 300
n = 500
K = 2
rho_true = np.array([0.9, 0.8])

# Creating the theta and eta matrices with given non-zero elements and coefficients
theta = np.zeros((q, K))
eta = np.zeros((p, K))

id_col1 = np.array([1, 6, 11, 16, 21]) - 1  # Adjusting for zero-indexing in Python
coefs_col1 = np.array([-2, -1, 0, 1, 2])
coefs_col2 = np.array([0, 0, 0, 1, 1])

theta[id_col1, 0] = coefs_col1
theta[id_col1, 1] = coefs_col2
theta[:, 0] /= np.sqrt(np.sum(theta[:, 0]**2))
theta[:, 1] /= np.sqrt(np.sum(theta[:, 1]**2))
eta = theta

# Identity covariance matrices for X and Y
sigma_X = np.identity(p)
sigma_Y = np.identity(q)

# Constructing the cross-covariance matrix sigma_YX
sigma_YX = sum(rho_true[k] * sigma_Y.dot(np.outer(theta[:, k], eta[:, k])).dot(sigma_X) for k in range(K))

# Assembling the full covariance matrix sigma
sigma = np.zeros((p + q, p + q))
sigma[:q, :q] = sigma_Y
sigma[q:, q:] = sigma_X
sigma[:q, q:] = sigma_YX
sigma[q:, :q] = sigma_YX.T

# Eigenvalue correction if necessary
eigvals, eigvecs = np.linalg.eigh(sigma)
# Set any small negative eigenvalues to zero (or a small positive value)
eigvals[eigvals < 0] = 0
# Reconstruct the corrected covariance matrix
sigma_corrected = eigvecs @ np.diag(eigvals) @ eigvecs.T

# Verify the matrix is now PSD
eigvals_corrected = np.linalg.eigvalsh(sigma_corrected)
