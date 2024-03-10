import numpy as np
import itertools
from functools import reduce
import time
import einsum

def reduce_mult(L):
    return reduce(lambda x, y: x*y, L)

#A = np.array([[1,4,1,7], [8,1,2,2], [7,4,3,4]])
#A.shape
# (3, 4)
# 
# A = np.array([[[1., 0., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 1., 0., 0., 0.]],
# 
#        [[0., 2., 0., 0., 0.],
#         [0., 0., 2., 0., 0.],
#         [0., 0., 0., 2., 0.],
#         [0., 0., 0., 0., 2.]],
# 
#        [[0., 0., 0., 0., 3.],
#         [0., 0., 0., 3., 0.],
#         [0., 0., 3., 0., 0.],
#         [0., 3., 0., 0., 0.]]])
# #print(A.shape)
# 
# B = np.array([[2,5], [0,1], [5,7], [9,2]])
#print(B.shape)
# (4, 2)

seedval = 42
np.random.seed(seedval)

A = np.random.rand(12,20,8,4)
B = np.random.rand(12,8,9,10)
#C = np.einsum('ij,jk->ki', A, B)

#A = np.array([1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 
#0.0, 2.0, 0.0, 0.0, 0.0,  0.0, 0.0, 2.0, 0.0, 0.0,  0.0, 0.0, 0.0, 2.0, 0.0,  0.0, 0.0, 0.0, 0.0, 2.0,
#0.0, 0.0, 0.0, 0.0, 3.0,  0.0, 0.0, 0.0, 3.0, 0.0,  0.0, 0.0, 3.0, 0.0, 0.0,  0.0, 3.0, 0.0, 0.0, 0.0],dtype=np.float64).reshape(3,4,5)
#B = np.array([2,5, 0,1, 5,7, 9,2],dtype=np.float64).reshape(4,2)

inputs = [A,B]

expr = 'ijka,ikbc->abcj'
#expr = 'ijl,jk->kli'
#print(expr)

start_time = time.time()

qry_expr, res_expr = expr.split('->')
inputs_expr = qry_expr.split(',')
inputs_expr, res_expr
#(['ij', 'jk'], 'ki')

keys = set([(key, size) for keys, input in zip(inputs_expr, inputs) for key, size in list(zip(keys, input.shape))])
#{('i', 3), ('j', 4), ('k', 2)}

sizes = dict(keys)
#{'i': 3, 'j': 4, 'k': 2}

ranges = [range(size) for _, size in keys]
#[range(0, 2), range(0, 3), range(0, 4)]

to_key = sizes.keys()
#['k', 'i', 'j']  #dict_keys(['k', 'j', 'i']))

domain = itertools.product(*ranges)

res = np.zeros([sizes[key] for key in res_expr])

for indices in domain:
    vals = {k: v for v, k in zip(indices, to_key)}
    #print(vals)
    res_ind = tuple(zip([vals[key] for key in res_expr]))
    inputs_ind = [tuple(zip([vals[key] for key in expr])) for expr in inputs_expr]
    #print(inputs_ind, res_ind)
    res[res_ind] += reduce_mult([M[i] for M, i in zip(inputs, inputs_ind)])

end_time = time.time()

C1 = res

execution_time = end_time - start_time

print(f"Execution time: {execution_time:.6f} seconds")

start_time2 = time.time()
C2 = np.einsum(expr, A,B)
end_time2 = time.time()
execution_time2 = end_time2 - start_time2

print(f"Execution time: {execution_time2:.6f} seconds")


start_time3 = time.time()
C3 = einsum.c_einsum(expr, A,B)
end_time3 = time.time()
execution_time3 = end_time3 - start_time3

print(f"Execution time: {execution_time3:.6f} seconds")

print(C1[0][0][0][0])
print(C2[0][0][0][0])
print(C3[0][0][0][0])