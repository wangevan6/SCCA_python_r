#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:44:15 2024

@author: water
"""

import numpy as np

def sweep(matrix, margin, stats, operation='/'):
    """
    Python equivalent of R's sweep function without relying on broadcasting.
    
    Parameters:
    - matrix: 2D NumPy array to be "swept".
    - margin: 0 for row-wise operation, 1 for column-wise operation.
    - stats: 1D NumPy array containing values to be used for the operation.
    - operation: The operation to perform; defaults to '/' (division).
    
    Returns:
    - The "swept" matrix after applying the operation.
    """
    result_matrix = np.empty_like(matrix)
    
    if margin == 1:  # Column-wise operation
        for i in range(matrix.shape[1]):  # Iterate over columns
            if operation == '/':
                result_matrix[:, i] = matrix[:, i] / stats[i]
            elif operation == '*':
                result_matrix[:, i] = matrix[:, i] * stats[i]
            # Add more operations as needed
    elif margin == 0:  # Row-wise operation
        for i in range(matrix.shape[0]):  # Iterate over rows
            if operation == '/':
                result_matrix[i, :] = matrix[i, :] / stats[i]
            elif operation == '*':
                result_matrix[i, :] = matrix[i, :] * stats[i]
            # Add more operations as needed
    else:
        raise ValueError("Unsupported margin. Use 0 for row-wise or 1 for column-wise operations.")
    
    return result_matrix

'''


# Assuming the sweep function is already defined as provided in the previous response

# Step 1: Create a 2D NumPy array
data_matrix = np.array([[1, 2], [3, 4], [5, 6]])

# Step 2: Define scaling factors (one for each column)
scaling_factors = np.array([2, 3])  # Scale the first column by 2, the second by 3

# Step 3: Apply the sweep function for column-wise division
swept_matrix = sweep(data_matrix, margin=1, stats=scaling_factors, operation='/')

# Step 4: Manual validation
expected_result = np.array([[1/2, 2/3], [3/2, 4/3], [5/2, 6/3]])

# Print results
print("Swept (scaled) matrix:\n", swept_matrix)
print("Expected result:\n", expected_result)
'''