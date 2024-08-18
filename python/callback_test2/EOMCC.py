import numpy as np

# Example matrix for demonstration
matrix = np.array([[1.0, 2.0, 3.0, 4.0],
                   [2.0, 3.0, 4.0, 5.0],
                   [3.0, 4.0, 5.0, 6.0],
                   [4.0, 5.0, 6.0, 7.0]])

def matvec(vec):
    return matrix*vec

def get_diagonal_element(index):
    # Assuming 'matrix' is a square matrix
    if index < 0 or index >= min(matrix.shape):
        raise IndexError("Index out of bounds")
    
    return matrix[index, index]

def get_global_shape():
    return matrix.shape