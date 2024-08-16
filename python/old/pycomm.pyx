#distutils: language = c++
from Comm cimport Comm, MKLComm
from libc.stdlib cimport free

cdef class mkl_comm:
    cdef MKLComm *comm_ptr

    def __cinit__(self):
        self.comm_ptr = new MKLComm(0, 1)

    def __dealloc__(self):
        free(self.comm_ptr)

    def clone(self):
        cdef mkl_comm new_comm = mkl_comm()
        new_comm.comm_ptr = <MKLComm*>self.comm_ptr.clone()
        return new_comm