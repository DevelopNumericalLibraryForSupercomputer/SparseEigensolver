# cython: language_level=3
# distutils: language = c++
from libcpp.memory cimport unique_ptr
#from Type cimport DEVICETYPE

cdef extern from "../include/Comm.hpp" namespace "SE":
    cdef cppclass Comm[device]:
        Comm() except+
        Comm(int rank, int world_size) except+

        void finalize()
        void barrier()
        Comm[device]* clone()
        void allgatherv[DATATYPE](DATATYPE* src, int sendcount, DATATYPE* trg, int* recvcounts)
        void scatterv[DATATYPE](DATATYPE* src, int* sendcounts, DATATYPE* trg, int recvcount, int root)
        void alltoallv[DATATYPE](DATATYPE* src, int* sendcounts, DATATYPE* trg, int* recvcounts)
        int get_rank()
        int get_world_size()
        int rank
        int world_size
        int count


cdef extern from "../include/device/mkl/MKLComm.hpp" namespace "SE":
    cdef cppclass MKLComm "SE::Comm<SE::DEVICETYPE::MKL>":
        MKLComm() except+
        MKLComm(int rank, int world_size) except+

        void finalize()
        void barrier()
        MKLComm* clone()
        void allgatherv[DATATYPE](DATATYPE* src, int sendcount, DATATYPE* trg, int* recvcounts)
        void scatterv[DATATYPE](DATATYPE* src, int* sendcounts, DATATYPE* trg, int recvcount, int root)
        void alltoallv[DATATYPE](DATATYPE* src, int* sendcounts, DATATYPE* trg, int* recvcounts)
        int get_rank()
        int get_world_size()
        int rank
        int world_size
        int count