# distutils: language = c++
from libcpp.memory cimport unique_ptr

cdef extern from "Comm.hpp" namespace "SE":
    cdef cppclass Comm[DEVICETYPE]:
        pass
        #Comm() except+
        #Comm(size_t rank, size_t world_size) except+
        #Comm[DEVICETYPE]* clone()

cdef extern from "device/mkl/MKLComm.hpp" namespace "SE":
    unique_ptr[MKLComm] create_comm(int argc, char *argv[])
    cdef cppclass MKLComm "SE::Comm<SE::DEVICETYPE::MKL>":
        MKLComm() except+
        #MKLComm* clone()
        MKLComm(const MKLComm& other) except+
        #copy assign operator
        MKLComm& operator=(const MKLComm& other) except+

        size_t get_rank()
        size_t get_world_size()
