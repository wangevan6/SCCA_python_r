#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:29:06 2024

@author: oujakusui
"""

import numpy as np
from sklearn.linear_model import Lasso

def SCCA_solution(x, y, x_Omega, y_Omega, alpha0, beta0, lambda_alpha, lambda_beta, niter=100, eps=1e-4):
    n, p = x.shape
    q = y.shape[1]

    for i in range(niter):
        x0 = np.dot(x_Omega, beta0)
        
        m_alpha = Lasso(alpha=lambda_alpha, fit_intercept=False, normalize=False)
        m_alpha.fit(y, x0)
        alpha1 = m_alpha.coef_
        
        if np.sum(np.abs(alpha1)) < eps:
            alpha0 = np.zeros(q)
            break
        
        id_nz = np.where(alpha1 != 0)[0]
        alpha1_scale = np.dot(y[:, id_nz], alpha1[id_nz])
        
        alpha1 = alpha1 / np.sqrt(np.dot(alpha1_scale.T, alpha1_scale) / (n - 1))
        
        y0 = np.dot(y_Omega, alpha1)
        
        m_beta = Lasso(alpha=lambda_beta, fit_intercept=False, normalize=False)
        m_beta.fit(x, y0)
        beta1 = m_beta.coef_
        
        if np.sum(np.abs(beta1)) < eps:
            beta0 = np.zeros(p)
            break
        
        id_nz = np.where(beta1 != 0)[0]
        beta1_scale = np.dot(x[:, id_nz], beta1[id_nz])
        
        beta1 = beta1 / np.sqrt(np.dot(beta1_scale.T, beta1_scale) / (n - 1))
        
        if np.sum(np.abs(alpha1 - alpha0)) < eps and np.sum(np.abs(beta1 - beta0)) < eps:
            break
        
        alpha0 = alpha1
        beta0 = beta1

    return {'alpha': alpha0, 'beta': beta0, 'niter': i + 1}

# Example usage
# SCCA_solution(x=x, y=y, x_Omega=x_tmp, y_Omega=y_tmp, alpha0=alpha0, beta0=beta0, lambda_alpha=lambda_alpha0, lambda_beta=lambda_beta0, niter=niter, eps=eps)
