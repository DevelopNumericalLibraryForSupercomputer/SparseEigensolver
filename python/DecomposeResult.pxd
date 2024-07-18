#distutils: language = c++
from libcpp.memory cimport unique_ptr

cdef extern from "decomposition/DecomposeResult.hpp" namespace "SE":
    cdef cppclass DecomposeResult[DATATYPE]:
        pass

    cdef cppclass DecomposeResult_double "SE::DecomposeResult<double>":
        DecomposeResult_double(size_t num_eig, unique_ptr[double[]] real_eigvals, unique_ptr[double[]] imag_eigvals) except+
