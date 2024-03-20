from TensorOperations cimport TensorOperations
from libcpp.memory cimport unique_ptr
from array cimport array, two
from Contiguous1DMap cimport Contiguous1DMap
from DenseTensor cimport DenseTensor2D
from DecomposeResult cimport DecomposeResult

from libcpp.string cimport string

#cdef extern from *:
#    ctypedef size_t BASE    "0"
#    ctypedef size_t MKL     "1"
#    ctypedef size_t MPI     "2"
#    ctypedef size_t CUDA    "11"
#    ctypedef size_t NCCL    "12"

#cdef extern from "../include/decomposition/IterativeSolver.hpp" namespace "SE":
#    cdef unique_ptr[DecomposeResult[datatype]] davidson[datatype, maptype, devicetype](TensorOperations* operations, DenseTensor2D* eigvec) except+

cdef extern from "../include/decomposition/Decompose.hpp" namespace "SE":
    #unique_ptr[DecomposeResult[double]] decompose(DenseTensor2D& tensor, DenseTensor2D* eigvec, string method)    
    #cdef unique_ptr[DecomposeResult[datatype]] decompose[datatype, maptype, device](SparseTensor2D& tensor, DenseTensor2D* eigvec, string method):
    #    pass
    unique_ptr[DecomposeResult[double]] decompose(TensorOperations* operations, DenseTensor2D* eigvec, string method)