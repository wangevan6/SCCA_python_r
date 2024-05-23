#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:47:35 2024

@author: oujakusui
"""

import numpy as np
from sklearn.cross_decomposition import CCA

# Sample data
X = np.array([[0.2, 0.4, 0.6], [0.3, 0.8, 0.5], [0.7, 0.6, 0.1], [0.5, 0.2, 0.9]])
Y = np.array([[1.0, 0.5], [0.6, 0.8], [0.3, 0.9], [0.8, 0.2]])

# Standardize the data
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)

# Perform CCA
cca = CCA(n_components=2)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)

# Canonical correlations
canonical_correlations = np.corrcoef(X_c.T, Y_c.T)[0, 1]
print("Canonical correlations:", canonical_correlations)

# Canonical weights
print("X canonical weights:", cca.x_weights_)
print("Y canonical weights:", cca.y_weights_)
