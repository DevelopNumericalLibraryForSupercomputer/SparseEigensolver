# distutils: language = c++
#Comm.pxd
from libcpp.memory cimport unique_ptr
#from Type cimport DEVICETYPE
cdef extern from "Type.hpp" namespace "SE":
    cdef enum class DEVICETYPE:
        BASE=0, 
        MKL=1,
        MPI=2,
        CUDA=11,
        NCCL=12


cdef extern from "Comm.hpp" namespace "SE":
    cdef cppclass Comm[DEVICETYPE]:
        Comm(size_t rank, size_t world_size)
        #Comm[DEVICETYPE]* clone()

cdef extern from "device/mkl/MKLComm.hpp" namespace "SE":
    cdef cppclass MKLComm:
        pass
        #MKLComm* clone()
#    # Comm<DEVICETYPE>* create_comm(int argc, char *argv[])
#    unique_ptr[Comm[DEVICETYPE]] create_comm(int argc, char *argv[])