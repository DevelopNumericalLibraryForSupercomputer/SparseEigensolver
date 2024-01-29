import numpy as np
import itertools
from functools import reduce

def reduce_mult(L):
    return reduce(lambda x, y: x*y, L)

#A = np.array([[1,4,1,7], [8,1,2,2], [7,4,3,4]])
#A.shape
# (3, 4)

A = np.array([[[1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0.]],

       [[0., 2., 0., 0., 0.],
        [0., 0., 2., 0., 0.],
        [0., 0., 0., 2., 0.],
        [0., 0., 0., 0., 2.]],

       [[0., 0., 0., 0., 3.],
        [0., 0., 0., 3., 0.],
        [0., 0., 3., 0., 0.],
        [0., 3., 0., 0., 0.]]])
print(A.shape)



B = np.array([[2,5], [0,1], [5,7], [9,2]])
print(B.shape)
# (4, 2)

#C = np.einsum('ij,jk->ki', A, B)

inputs = [A,B]

expr = 'ijl,jk->kli'
print(expr)
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
    print(vals)
    res_ind = tuple(zip([vals[key] for key in res_expr]))
    inputs_ind = [tuple(zip([vals[key] for key in expr])) for expr in inputs_expr]
    print(inputs_ind, res_ind)
    res[res_ind] += reduce_mult([M[i] for M, i in zip(inputs, inputs_ind)])

print(res)

print(np.einsum(expr, A,B))
