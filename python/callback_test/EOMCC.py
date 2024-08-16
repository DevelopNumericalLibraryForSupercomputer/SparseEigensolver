#EOMCC.py
import numpy as np

def matvec(arr):
    indices = np.arange(1, arr.shape[1] + 1)
    result = arr * indices
    return result

# 예시 사용법
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
new_arr = multiply_columns_by_index(arr)
print(new_arr)