#Type.pxd
cdef extern from "Type.hpp" namespace "SE":
    cdef enum DEVICETYPE:
        BASE = 0
        MKL  = 1
        MPI  = 2
        CUDA = 11
        NCCL = 12

    #cdef enum MTYPE:
    #    Contiguous1D=0,
    #    placeholder=1
        
    #cdef enum STORETYPE:
    #    DENSE=0,
    #    COO=1

