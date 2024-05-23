#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:28:11 2024

@author: oujakusui
"""

import numpy as np
from numpy.linalg import svd, norm, solve
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from SCCA_main_solutionV4 import SCCA
from scca_init0 import init0

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


# Compute the square root of the corrected sigma using SVD
U, s, Vh = np.linalg.svd(sigma_corrected)
sigma_sqrt_corrected = U @ np.diag(np.sqrt(s)) @ Vh

# Set the random seed and generate the data using the corrected sigma
np.random.seed(314159)
Z_corrected = np.random.randn(2 * n * (p + q)).reshape(2 * n, p + q) @ sigma_sqrt_corrected
Y_corrected = Z_corrected[:, :q]
X_corrected = Z_corrected[:, q:(p + q)]

# Split the data into training and tuning sets
id_train = np.random.choice(2 * n, n, replace=False)
X_train = X_corrected[id_train, :]
Y_train = Y_corrected[id_train, :]
X_tune = X_corrected[~id_train, :]
Y_tune = Y_corrected[~id_train, :]

# Center the training data
X_train -= X_train.mean(axis=0)
Y_train -= Y_train.mean(axis=0)

# The rest of the code for SCCA analysis is not executed here as it requires the 'SCCA' and 'init0' functions
# which are not provided and appear to be custom functions for Sparse Canonical Correlation Analysis.

# However, we can provide the norm differences based on provided theta and eta
from numpy.linalg import norm, solve

alpha_hat = np.random.randn(q, K)  # Placeholder, replace with actual SCCA result
beta_hat = np.random.randn(p, K)   # Placeholder, replace with actual SCCA result

# Calculate norm differences using the Frobenius norm for matrices
alpha_norm_diff = norm(alpha_hat @ np.linalg.pinv(alpha_hat.T @ alpha_hat) @ alpha_hat.T - theta @ np.linalg.pinv(theta.T @ theta) @ theta.T, 'fro')
beta_norm_diff = norm(beta_hat @ np.linalg.pinv(beta_hat.T @ beta_hat) @ beta_hat.T - eta @ np.linalg.pinv(eta.T @ eta) @ eta.T, 'fro')

print(alpha_norm_diff, beta_norm_diff)

