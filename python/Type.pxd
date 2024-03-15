cdef extern from "../include/Type.hpp" namespace "SE":
    cdef enum DEVICETYPE:
        BASE = 0
        MKL = 1
        MPI = 2
        CUDA = 11
        NCCL = 12