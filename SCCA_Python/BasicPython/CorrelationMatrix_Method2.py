#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:46:38 2024

@author: oujakusui
"""

import numpy as np
from scipy.stats import pearsonr

# Sample data: replace these with your actual data
x = np.random.rand(100, 5)  # 100 observations of 5-dimensional vector
y = np.random.rand(100, 3)  # 100 observations of 3-dimensional vector

# Initialize an empty correlation matrix
correlation_matrix_existing = np.zeros((x.shape[1], y.shape[1]))

# Compute pairwise correlations
for i in range(x.shape[1]):
    for j in range(y.shape[1]):
        correlation_matrix_existing[i, j] = pearsonr(x[:, i], y[:, j])[0]

print("Correlation matrix using existing methods:")
print(correlation_matrix_existing)
