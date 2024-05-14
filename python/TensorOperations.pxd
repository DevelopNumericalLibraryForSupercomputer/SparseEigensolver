from array cimport array, two
from DenseTensor cimport DenseTensor, DenseTensor1D, DenseTensor2D

cdef extern from "../include/decomposition/TensorOperations.hpp" namespace "SE":
    cdef cppclass TensorOperations:
        TensorOperations() except +

        DenseTensor1D matvec(const DenseTensor1D& vec)
        DenseTensor2D matvec(const DenseTensor2D& vec)
        double get_diag_element(const int index)
        array[int,two] get_global_shape()
