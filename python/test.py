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