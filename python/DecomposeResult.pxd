from libcpp.vector cimport vector

cdef extern from "../include/decomposition/DecomposeResult.hpp" namespace "SE":
    cdef cppclass DecomposeResult[datatype]:
        DecomposeResult() except+
        DecomposeResult(const size_t num_eig, vector[datatype] real_eigvals, vector[datatype] real_eigvals) except+
        
        const size_t num_eig
        vector[datatype] real_eigvals
        vector[datatype] imag_eigvals
