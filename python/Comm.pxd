# distutils: language = c++
from libcpp.memory cimport unique_ptr

cdef extern from "Comm.hpp" namespace "SE":
    cdef cppclass Comm[DEVICETYPE]:
        pass
        #Comm() except+
        #Comm(size_t rank, size_t world_size) except+
        #Comm[DEVICETYPE]* clone()

cdef extern from "device/mkl/MKLComm.hpp" namespace "SE":
    cdef cppclass MKLComm "SE::Comm<SE::DEVICETYPE::MKL>":
        MKLComm() except+
        MKLComm(size_t rank, size_t world_size) except+
        MKLComm* clone()
    

    