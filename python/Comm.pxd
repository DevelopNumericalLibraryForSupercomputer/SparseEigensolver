# cython: language_level=3
# distutils: language = c++
from libcpp.memory cimport unique_ptr
#from Type cimport DEVICETYPE

cdef extern from "../include/Comm.hpp" namespace "SE":
    cdef cppclass Comm[device]:
        Comm() except+
        Comm(size_t rank, size_t world_size) except+

        void finalize()
        void barrier()
        Comm[device]* clone()
        void allgatherv[DATATYPE](DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t* recvcounts)
        void scatterv[DATATYPE](DATATYPE* src, size_t* sendcounts, DATATYPE* trg, size_t recvcount, size_t root)
        void alltoallv[DATATYPE](DATATYPE* src, size_t* sendcounts, DATATYPE* trg, size_t* recvcounts)
        size_t get_rank()
        size_t get_world_size()
        size_t rank
        size_t world_size
        size_t count


cdef extern from "../include/device/mkl/MKLComm.hpp" namespace "SE":
    cdef cppclass MKLComm "SE::Comm<SE::DEVICETYPE::MKL>":
        MKLComm() except+
        MKLComm(size_t rank, size_t world_size) except+

        void finalize()
        void barrier()
        MKLComm* clone()
        void allgatherv[DATATYPE](DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t* recvcounts)
        void scatterv[DATATYPE](DATATYPE* src, size_t* sendcounts, DATATYPE* trg, size_t recvcount, size_t root)
        void alltoallv[DATATYPE](DATATYPE* src, size_t* sendcounts, DATATYPE* trg, size_t* recvcounts)
        size_t get_rank()
        size_t get_world_size()
        size_t rank
        size_t world_size
        size_t count