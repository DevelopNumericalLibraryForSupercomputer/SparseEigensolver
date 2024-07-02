#distutils: language = c++
#from Type cimport DEVICETYPE
from Comm cimport Comm, DEVICETYPE, MKLComm

#from libcpp.memory cimport unique_ptr

#wrapper of Comm<DEVICETYPE::MKL>
cdef class mkl_comm:
    #cdef unique_ptr[Comm[DEVICETYPE]] comm_ptr
    cdef MKLComm *comm_ptr

    def __cinit__(self):
        #Constructor : Comm(size_t rank, size_t world_size)
        self.comm_ptr = new Comm[DEVICETYPE](0, 1)

    def __dealloc__(self):
        pass

    #def clone(self):
    #    cdef Comm* new_comm_ptr = self.comm_ptr.get().clone()
    #    cdef mkl_comm new_wrapper = mkl_comm.__new__(mkl_comm)
    #    self.comm_ptr = unique_ptr[Comm[DEVICETYPE]](new_comm_ptr)
    #    return new_wrapper

    
