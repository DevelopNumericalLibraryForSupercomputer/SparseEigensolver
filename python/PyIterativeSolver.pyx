from array cimport array, two
from Contiguous1DMap cimport Contiguous1DMap
from Comm cimport Comm, MKLComm
from DenseTensor cimport DenseTensor2D, DenseTensor1D
#from Type cimport DEVICE_MKL
from libcpp cimport bool
from TensorOperations cimport TensorOperations
from DecomposeResult cimport DecomposeResult
from Decompose cimport decompose

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
np.import_array()


cdef call_decompose(TensorOperations* operations, np.ndarray[double, ndim=2] vecs):
    #ndarray to DenseTensor2D
    print("start call_decompose")
    cdef array[size_t, two] global_shape
    cdef array[bool, two] is_parallel
    is_parallel[0] = True
    is_parallel[1] = False
    global_shape[0] = vecs.shape[0]
    global_shape[1] = vecs.shape[1]
    print("MKLCOMM")
    cdef MKLComm comm = MKLComm(0,1)
    cdef size_t my_rank = comm.get_rank()
    cdef size_t global_size = comm.get_world_size()
    cdef Contiguous1DMap[two] map = Contiguous1DMap[two](global_shape, my_rank, global_size, is_parallel)
    cdef DenseTensor2D* c_vecs = new DenseTensor2D(comm, map, &vecs[0,0])
    print("before decompose")

    #call davidson
    cdef unique_ptr[DecomposeResult[double]] c_result = decompose(operations, c_vecs, "davidson")
    print("after decompose")
    #process DenseTensorResult
    #cdef size_t num_eig = c_result.get().num_eig
    #cdef new_real_eigvals = np.zeros(num_eig)
    #print(num_eig)
    cdef vector[double] c_real_eigvals = c_result.get().real_eigvals
    print(c_real_eigvals[0])
    
cdef extern from "../include/decomposition/TestOperations.hpp" namespace "SE":
    cdef cppclass TestTensorOperations(TensorOperations):
        TestTensorOperations() except +
        TestTensorOperations(size_t n) except +

        DenseTensor1D matvec(const DenseTensor1D& vec)
        DenseTensor2D matvec(const DenseTensor2D& vec)
        double get_diag_element(const size_t index)
        array[size_t,two] get_global_shape()


def testrun(np.ndarray[double, ndim=2] vecs):
    cdef TensorOperations* a = new TestTensorOperations(3)
    print("a")
    call_decompose(a, vecs)
