#distutils: language = c++
# PyDecompose.pyx
# Decompose.hpp의 Cython wrapper

import numpy as np
cimport numpy as np
np.import_array()
#from cpython cimport PyObject, Py_INCREF, Py_DECREF
#from libcpp cimport size_t
from libcpp.memory cimport unique_ptr
#from libcpp.string cimport string

from cython.view cimport array as cvarray

from Comm cimport MKLComm, create_comm
from array cimport array, one, two, three, four, five, six
from Contiguous1DMap cimport Contiguous1DMap
from DenseTensor cimport MKL_DenseTensor2D
from SparseTensor cimport MKL_SparseTensor2D
from Decompose cimport decompose_dense
from DecomposeResult cimport DecomposeResult_double
#from PyDecomposeResult cimport PyDecomposeResult
#from PyOperations cimport Tensor_Operations

from libc.stdlib cimport malloc, free

cpdef c_decompose(np.ndarray tensor, np.ndarray eigvec):
    assert tensor.ndim == 2
    
    #cdef unique_ptr[MKLComm] comm = create_comm(0, NULL)

    cdef MKLComm comm = MKLComm()
    cdef array[size_t, two] tensor_shape
    tensor_shape[0] = tensor.shape[0]
    tensor_shape[1] = tensor.shape[1]
    #print(tensor_shape[0])
    #print(tensor_shape[1])
    
    cdef Contiguous1DMap[two] tensor_map = Contiguous1DMap[two](tensor_shape, 0, 1)
    cdef double* tensor_ptr = <double*> tensor.data
    cdef MKL_DenseTensor2D c_tensor = MKL_DenseTensor2D(comm, tensor_map, tensor_ptr)

    cdef array[size_t, two] eigvec_shape
    eigvec_shape[0] = eigvec.shape[0] # row
    eigvec_shape[1] = eigvec.shape[1] # col
    cdef Contiguous1DMap[two] eigvec_map = Contiguous1DMap[two](eigvec_shape, 0, 1)
    #cdef double* eigvec_ptr = <double*> eigvec.data
    #cdef MKL_DenseTensor2D* c_eigvec = new MKL_DenseTensor2D(comm, eigvec_map, eigvec_ptr)
    #
    #cdef unique_ptr[DecomposeResult_double] result = decompose_dense(c_tensor, c_eigvec, method)
    #
    ##copy c_eigvec.data into np.ndarray
    #cdef double* c_eigvec_data = c_eigvec.data
    #eigvec[...] = np.frombuffer(c_eigvec_data, dtype=np.double, count=eigvec_shape[0] * eigvec_shape[1]).reshape((eigvec_shape[0], eigvec_shape[1]))

    #

    ##cdef PyDecomposeResult py_result = PyDecomposeResult(result)
    #return True
    

    #memoryview on a c array example
    cdef int size = tensor.shape[0]
    
    cdef double* new_eigvec
    new_eigvec = <double*> malloc(2*size * sizeof(double))

    for i in range(2*size):
        new_eigvec[i] = i

    print(new_eigvec[1])
    cdef double[:] eigvecs_view = <double[:2*size]> new_eigvec
    print("before assign")
    new_np_eigvec = np.asarray(eigvecs_view).reshape((2,size))
    print(new_np_eigvec)
    print("after assign")
    free(new_eigvec)
    return new_np_eigvec




#def c_decompose(Tensor_Operations py_operations, np.ndarray eigvec, string method):
#    cdef double* eigvec_ptr = <double*> eigvec.data
#
#    if eigvec.ndim == 1:
#        return decompose[double, MAPTYPE, DEVICETYPE](<TensorOperations*> &py_operations._operations, <double*> eigvec_ptr, eigvec.shape[0], method)
#    elif eigvec.ndim == 2:
#        return decompose[double, MAPTYPE, DEVICETYPE](<TensorOperations*> &py_operations._operations, <double*> eigvec_ptr, eigvec.shape[0], eigvec.shape[1], method)
#    else:
#        raise ValueError("eigvec must be 1D or 2D array")


#def c_decompose(Tensor_Operations py_operations, np.ndarray eigvec, string method):
#    cdef double* eigvec_ptr = <double*> eigvec.data
#    return davidson[double, MAPTYPE, DEVICETYPE](<TensorOperations*> &py_operations._operations, eigvec, method)