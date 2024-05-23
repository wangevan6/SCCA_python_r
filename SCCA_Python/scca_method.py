#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:19:32 2024

@author: oujakusui
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from scipy.linalg import svd
from scipy.stats import zscore



def SCCA(x, y, lambda_alpha, lambda_beta, niter, npairs, init_method, alpha_init=None, beta_init=None, standardize=True, eps=1e-4):
    p, q, n = x.shape[1], y.shape[1], x.shape[0]

    # Standardize x and y if required
    if standardize:
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=1)
        y = (y - np.mean(y, axis=0)) / np.std(y, axis=0, ddof=1)

    # Covariance matrices
    sigma_YX_hat = np.cov(y.T, x.T)[:q, q:]
    sigma_X_hat = np.cov(x.T)
    sigma_Y_hat = np.cov(y.T)

    alpha = np.zeros((q, npairs))
    beta = np.zeros((p, npairs))

    # Choose the first init method if multiple are provided
    if isinstance(init_method, list):
        init_method = init_method[0]

    # Initialize alpha and beta
    if alpha_current is None:
        alpha_current = np.zeros((q, 0))
        beta_current = np.zeros((p, 0))
        npairs0 = 0
    else:
        npairs0 = alpha_current.shape[1]
        alpha[:, :npairs0] = alpha_current
        beta[:, :npairs0] = beta_current


    if alpha_init is None:
        if alpha_current is None:
            init_results = init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, n)
            alpha_init = init_results["alpha_init"]
            beta_init = init_results["beta_init"]
        else:
            init_results = init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, npairs0, alpha_current, beta_current, n, eps)
            alpha_init = init_results["alpha_init"]
            beta_init = init_results["beta_init"]
    
    n_iter_converge = np.zeros(npairs - npairs0)

    # Iteratively find each pair of canonical variables
    for ipair in range(npairs0, npairs):
        omega = find_Omega(sigma_YX_hat, ipair + 1, alpha[:, :ipair], beta[:, :ipair], y, x)

        x_tmp = omega @ x
        y_tmp = omega.T @ y

        lambda_alpha0 = lambda_alpha[ipair - npairs0]
        lambda_beta0 = lambda_beta[ipair - npairs0]

        alpha0 = alpha_init[:, [ipair]]
        beta0 = beta_init[:, [ipair]]

        result = SCCA_solution(x, y, x_tmp, y_tmp, alpha0, beta0, lambda_alpha0, lambda_beta0, niter, eps)

        alpha[:, ipair] = result['alpha']
        beta[:, ipair] = result['beta']
        n_iter_converge[ipair - npairs0] = result['niter']

        if ipair + 1 < npairs and init_method == "sparse":
            init_results = init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, ipair + 1, alpha[:, :(ipair + 1)], beta[:, :(ipair + 1)], n, eps)
            alpha_init = init_results["alpha_init"]
            beta_init = init_results["beta_init"]
    
    return {"alpha": alpha, "beta": beta, "n_iter_converge": n_iter_converge}



def init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, n, d=None):
    p = sigma_X_hat.shape[1]
    q = sigma_Y_hat.shape[1]

    if init_method == "svd":
        U, _, Vt = svd(sigma_YX_hat)
        alpha_init = U[:, :npairs]
        beta_init = Vt.T[:, :npairs]
    elif init_method == "uniform":
        alpha_init = np.ones((q, npairs))
        beta_init = np.ones((p, npairs))
    elif init_method == "random":
        alpha_init = np.random.randn(q, npairs)
        beta_init = np.random.randn(p, npairs)
    elif init_method == "sparse":
        if d is None:
            d = np.sqrt(n)
        thresh = np.sort(np.abs(sigma_YX_hat).ravel())[::-1][int(d)-1]
        row_max = np.max(np.abs(sigma_YX_hat), axis=1)
        col_max = np.max(np.abs(sigma_YX_hat), axis=0)
        idx_row = row_max > thresh
        idx_col = col_max > thresh

        U, _, Vt = svd(sigma_YX_hat[idx_row][:, idx_col])
        alpha_init = np.zeros((q, npairs))
        beta_init = np.zeros((p, npairs))
        alpha_init[idx_row, 0] = U[:, 0]
        beta_init[idx_col, 0] = Vt.T[:, 0]

    alpha_scale = np.diag(alpha_init.T @ sigma_Y_hat @ alpha_init)
    beta_scale = np.diag(beta_init.T @ sigma_X_hat @ beta_init)
    alpha_init /= np.sqrt(alpha_scale)[:, np.newaxis]
    beta_init /= np.sqrt(beta_scale)[:, np.newaxis]

    return {"alpha_init": alpha_init, "beta_init": beta_init}


def init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, npairs0, alpha_current, beta_current, n, eps=1e-4, d=None):
    p = sigma_X_hat.shape[1]
    q = sigma_Y_hat.shape[1]
    alpha_init = np.zeros((q, 1))
    beta_init = np.zeros((p, 1))
    
    if init_method == "svd":
        U, _, Vt = svd(sigma_YX_hat)
        alpha_init = U[:, npairs0][:, np.newaxis]
        beta_init = Vt.T[:, npairs0][:, np.newaxis]
    elif init_method == "uniform":
        alpha_init = np.ones((q, 1))
        beta_init = np.ones((p, 1))
    elif init_method == "random":
        alpha_init = np.random.randn(q, 1)
        beta_init = np.random.randn(p, 1)
    elif init_method == "sparse":
        if d is None:
            d = np.sqrt(n)
        sigma_YX_tmp = sigma_YX_hat - sigma_Y_hat @ alpha_current @ (alpha_current.T @ sigma_YX_hat @ beta_current) @ beta_current.T @ sigma_X_hat
        thresh = np.sort(np.abs(sigma_YX_tmp).ravel())[::-1][int(d)-1]
        row_max = np.max(np.abs(sigma_YX_tmp), axis=1)
        col_max = np.max(np.abs(sigma_YX_tmp), axis=0)
        idx_row = row_max > thresh
        idx_col = col_max > thresh
        
        U, _, Vt = svd(sigma_YX_tmp[idx_row][:, idx_col])
        alpha_init[idx_row, 0] = U[:, 0]
        beta_init[idx_col, 0] = Vt.T[:, 0]
    
    alpha_scale = float(alpha_init.T @ sigma_Y_hat @ alpha_init)
    beta_scale = float(beta_init.T @ sigma_X_hat @ beta_init)
    alpha_init /= np.sqrt(alpha_scale)
    beta_init /= np.sqrt(beta_scale)
    
    return {"alpha_init": alpha_init, "beta_init": beta_init}


def find_Omega(sigma_YX_hat, npairs, alpha=None, beta=None, y=None, x=None):
    n = y.shape[0]
    if npairs > 1:
        rho = alpha.T @ sigma_YX_hat @ beta
        omega = np.eye(n) - y @ alpha @ rho @ beta.T @ x.T / n
    else:
        omega = np.eye(n)
    return omega


def SCCA_solution(x, y, x_Omega, y_Omega, alpha0, beta0, lambda_alpha, lambda_beta, niter=100, eps=1e-4):
    n = x.shape[0]
    q = y.shape[1]
    p = x.shape[1]
    
    for i in range(niter):
        x0 = x_Omega @ beta0
        lasso_alpha = Lasso(alpha=lambda_alpha, fit_intercept=False, normalize=False)
        lasso_alpha.fit(y, x0)
        alpha1 = lasso_alpha.coef_
        
        if np.sum(np.abs(alpha1)) < eps:
            alpha0 = np.zeros(q)
            break
        
        alpha1_scale = y[:, np.abs(alpha1) > eps] @ alpha1[np.abs(alpha1) > eps]
        alpha1 /= np.sqrt(alpha1_scale.T @ alpha1_scale / (n - 1))
        
        y0 = y_Omega @ alpha1
        lasso_beta = Lasso(alpha=lambda_beta, fit_intercept=False, normalize=False)
        lasso_beta.fit(x, y0)
        beta1 = lasso_beta.coef_
        
        if np.sum(np.abs(beta1)) < eps:
            beta0 = np.zeros(p)
            break
        
        beta1_scale = x[:, np.abs(beta1) > eps] @ beta1[np.abs(beta1) > eps]
        beta1 /= np.sqrt(beta1_scale.T @ beta1_scale / (n - 1))
        
        if np.sum(np.abs(alpha1 - alpha0)) < eps and np.sum(np.abs(beta1 - beta0)) < eps:
            break
        
        alpha0 = alpha1
        beta0 = beta1
    
    return {"alpha": alpha0, "beta": beta0, "niter": i+1}

def cv_SCCA_equal(x, y, lambda_vals, nfolds=5, alpha_init=None, beta_init=None, eps=1e-3, niter=20):
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr

    n = x.shape[0]
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=None)
    rho = np.zeros(len(lambda_vals))

    for i_lambda, lambda_val in enumerate(lambda_vals):
        rho_fold = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler_x = StandardScaler().fit(x_train)
            scaler_y = StandardScaler().fit(y_train)
            x_train = scaler_x.transform(x_train)
            y_train = scaler_y.transform(y_train)
            x_test = scaler_x.transform(x_test)
            y_test = scaler_y.transform(y_test)
            obj = SCCA(x_train, y_train, alpha_init=alpha_init, beta_init=beta_init, lambda_alpha=lambda_val, lambda_beta=lambda_val, eps=eps, niter=niter, standardize=False)  # Note standardize=False because we already standardized
            # Compute the correlation on the test set
            alpha, beta = obj['alpha'], obj['beta']
            cor = pearsonr(x_test @ beta, y_test @ alpha)[0]
            rho_fold.append(np.abs(cor))

        rho[i_lambda] = np.mean(rho_fold)

    best_lambda = lambda_vals[rho.argmax()]

    return {"rho": rho, "lambda": lambda_vals, "best_lambda": best_lambda}


def cv_SCCA(x, y, lambda_alpha, lambda_beta, nfolds=5, alpha_init=None, beta_init=None, eps=1e-3, niter=10, standardize=True):
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr

    n = x.shape[0]
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=None)
    rho = np.zeros((len(lambda_alpha), len(lambda_beta)))

    for i_lambda, la in enumerate(lambda_alpha):
        for j_lambda, lb in enumerate(lambda_beta):
            rho_fold = []
            for train_index, test_index in kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if standardize:
                    scaler_x = StandardScaler().fit(x_train)
                    scaler_y = StandardScaler().fit(y_train)
                    x_train = scaler_x.transform(x_train)
                    y_train = scaler_y.transform(y_train)
                    x_test = scaler_x.transform(x_test)
                    y_test = scaler_y.transform(y_test)

                obj = SCCA(x_train, y_train, alpha_init=alpha_init, beta_init=beta_init, lambda_alpha=la, lambda_beta=lb, eps=eps, niter=niter, standardize=False)  # Note standardize=False because we already standardized
                
                # Compute the correlation on the test set
                alpha, beta = obj['alpha'], obj['beta']
                cor = pearsonr(x_test @ beta, y_test @ alpha)[0]
                rho_fold.append(np.abs(cor))

            rho[i_lambda, j_lambda] = np.mean(rho_fold)

    # Identify the best lambda_alpha and lambda_beta
    best_indices = np.unravel_index(rho.argmax(), rho.shape)
    best_lambda_alpha = lambda_alpha[best_indices[0]]
    best_lambda_beta = lambda_beta[best_indices[1]]

    return {"rho": rho, "lambda_alpha": lambda_alpha, "lambda_beta": lambda_beta, "best_lambda_alpha": best_lambda_alpha, "best_lambda_beta": best_lambda_beta}
