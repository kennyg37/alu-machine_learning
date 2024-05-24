import numpy as np 

def determinant(matrix):
    """This function calculates the determinant of a matrix"""
    
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    
    return np.linalg.det(matrix)