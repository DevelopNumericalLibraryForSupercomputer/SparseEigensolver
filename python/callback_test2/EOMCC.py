import numpy as np

def matvec(arr):
    # Convert the input to a NumPy array if it isn't one already
    arr = np.array(arr)
    
    # Proceed with the original function logic
    indices = np.arange(1, arr.shape[1] + 1)
    result = arr * indices
    return result.tolist()  # Convert the result back to a list to return to C++
