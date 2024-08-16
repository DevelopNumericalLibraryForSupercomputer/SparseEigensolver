#import pycomm.pyx
#from PySparseTensor import mkl_comm
#from PySparseTensor import dense_tensor1d, dense_tensor2d

#print("Comm test")
#print("Create comm")
#new_comm = mkl_comm()
#print("Create Comm using create_comm")
#new_2comm = new_comm.clone()

#print("Dense Tensor test")
#print("Create dense tensor")
#new_tensor = dense_tensor1d(0, 1)
#new_tensor = dense_tensor2d(0, 1)
#print("Create Dense Tensor using create_dense_tensor")


from PyEigensolver import c_decompose
import numpy as np
mat = np.zeros((5,5))
guess = np.zeros((2,5))
for i in range(5):
    mat[i,i] = 1.0
    if i+1 < 5:
        mat[i,i+1] = -0.3
        mat[i+1,i] = -0.3

for i in range(2):
    guess[i,i] = 1.0
print("before dc")
print(mat)
print(guess)
a = c_decompose(mat, guess)
print("after dc")
print(mat)
print(a)


# example.py
from py_operations import PyTensorOperationsWrapper

def matrix_one_vec_callback(input_vec, output_vec, size, user_data):
    for i in range(size):
        output_vec[i] = input_vec[i] * 2

def matrix_mult_vec_callback(input_vecs, output_vec, num_vec, size, user_data):
    for i in range(num_vec):
        for j in range(size):
            output_vec[i * size + j] = input_vecs[i * size + j] * 3

def get_diag_element_callback(index, user_data):
    return float(index)

def get_global_shape_callback(shape, user_data):
    shape[0] = 10
    shape[1] = 5

# 콜백 함수 포인터를 전달하여 객체 생성
tensor_ops = PyTensorOperationsWrapper(matrix_one_vec_callback, matrix_mult_vec_callback, get_diag_element_callback, get_global_shape_callback, None)

# 텐서 연산 수행
result = tensor_ops.matvec(your_tensor)
print(result)