#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:10:11 2024

@author: oujakusui
"""
import numpy as np

y = np.array([[5,3,2,2,0],
             [0,0,0,0,0]])
alpha1 = np.array([0,1,0,0,1])
non_zero_indices = (alpha1 != 0).flatten() 
print(non_zero_indices)
alpha1_scale = y[:, non_zero_indices]