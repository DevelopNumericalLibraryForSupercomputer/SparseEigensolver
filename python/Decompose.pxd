from TensorOperations cimport TensorOperations
from libcpp.memory cimport unique_ptr
from array cimport array, two
from Contiguous1DMap cimport Contiguous1DMap
from DenseTensor cimport DenseTensor2D
from DecomposeResult cimport DecomposeResult

from libcpp.string cimport string

#cdef extern from *:
#    ctypedef int BASE    "0"
#    ctypedef int MKL     "1"
#    ctypedef int MPI     "2"
#    ctypedef int CUDA    "11"
#    ctypedef int NCCL    "12"

#cdef extern from "../include/decomposition/IterativeSolver.hpp" namespace "SE":
#    cdef unique_ptr[DecomposeResult[datatype]] davidson[datatype, maptype, devicetype](TensorOperations* operations, DenseTensor2D* eigvec) except+

cdef extern from "../include/decomposition/Decompose.hpp" namespace "SE":
    #unique_ptr[DecomposeResult[double]] decompose(DenseTensor2D& tensor, DenseTensor2D* eigvec, string method)    
    #cdef unique_ptr[DecomposeResult[datatype]] decompose[datatype, maptype, device](SparseTensor2D& tensor, DenseTensor2D* eigvec, string method):
    #    pass
    unique_ptr[DecomposeResult[double]] decompose(TensorOperations* operations, DenseTensor2D* eigvec, string method)