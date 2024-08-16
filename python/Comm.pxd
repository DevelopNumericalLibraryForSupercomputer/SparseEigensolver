# distutils: language = c++
from libcpp.memory cimport unique_ptr

cdef extern from "Comm.hpp" namespace "SE":
    cdef cppclass Comm[DEVICETYPE]:
        pass
    cdef cppclass MKLComm "SE::Comm<SE::DEVICETYPE::MKL>":
        pass
    cdef cppclass MPIComm "SE::Comm<SE::DEVICETYPE::MPI>":
        pass