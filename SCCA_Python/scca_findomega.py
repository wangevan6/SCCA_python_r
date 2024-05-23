#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:38:04 2024

@author: water
"""

import unittest
import numpy as np

def find_Omega(sigma_YX_hat, npairs, alpha=None, beta=None, y=None, x=None):
    n = y.shape[0]  # Equivalent to nrow in R
    if npairs > 1:
        rho = alpha.T @ sigma_YX_hat @ beta
        omega = np.eye(n) - y @ alpha @ rho @ beta.T @ x.T / n
    else:
        omega = np.eye(n)
    return omega


class TestFindOmega(unittest.TestCase):
    def test_multiple_pairs(self):
        # Define the matrices
        y = np.array([[1, 2], [3, 4]])
        x = np.array([[2, 1], [4, 3]])
        alpha = np.array([[0.5], [0.5]])
        beta = np.array([[0.5], [0.5]])
        sigma_YX_hat = np.array([[1, 0.5], [0.5, 1]])
        npairs = 2
        
        # Calculate expected outcome
        n = y.shape[0]
        rho = alpha.T @ sigma_YX_hat @ beta
        expected_omega = np.eye(n) - y @ alpha @ rho @ beta.T @ x.T / n
        
        # Test the function
        result_omega = find_Omega(sigma_YX_hat, npairs, alpha, beta, y, x)
        np.testing.assert_array_almost_equal(result_omega, expected_omega)

    def test_single_pair(self):
        # Define the matrices
        y = np.array([[1, 2], [3, 4]])
        npairs = 1
        
        # Expected outcome should be an identity matrix
        expected_omega = np.eye(y.shape[0])
        
        # Test the function
        result_omega = find_Omega(None, npairs, None, None, y, None)
        np.testing.assert_array_almost_equal(result_omega, expected_omega)

# Run the tests
if __name__ == '__main__':
    unittest.main()
