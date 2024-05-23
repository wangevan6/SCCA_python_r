#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:19:55 2024

@author: oujakusui
"""

import numpy as np

np.random.seed(42)

x = np.random.rand(5,100)
y = np.random.rand(5,100)
print(x)
print(y)

x_centered = x - np.mean(x, axis= 0)
y_centered = y - np.mean(x, axis= 0)

print(x_centered)
print(y_centered)

std_x = np.std(x_centered, ddof = 1 )
std_y = np.std(y_centered, ddof = 1)

# To be finished .....