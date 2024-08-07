#distutils: language = c++
from Comm cimport Comm, MKLComm
from Contiguous1DMap cimport Map, Contiguous1DMap
from DenseTensor cimport MKL_DenseTensor1D, MKL_DenseTensor2D
from array cimport array, one, two, three, four, five, six
from libc.stdlib cimport free

#cimport numpy as cpython
#import numpy as np

cdef class dense_tensor1d:
    cdef:
        MKL_DenseTensor1D* tensor_ptr
        MKLComm comm
        Contiguous1DMap[one] map

    def __cinit__(self, size_t rank, size_t world_size):
        self.comm = MKLComm(rank, world_size)
        self.map = Contiguous1DMap[one](array[size_t, one](), rank, world_size)
        self.tensor_ptr = new MKL_DenseTensor1D(self.comm, self.map)

    def __dealloc__(self):
        free(self.tensor_ptr)

cdef class dense_tensor2d:
    cdef:
        MKL_DenseTensor2D* tensor_ptr
        MKLComm comm
        Contiguous1DMap[two] map

    def __cinit__(self, size_t rank, size_t world_size):
        self.comm = MKLComm(rank, world_size)
        self.map = Contiguous1DMap[two](array[size_t, two](), rank, world_size)
        self.tensor_ptr = new MKL_DenseTensor2D(self.comm, self.map)

    def __dealloc__(self):
        free(self.tensor_ptr)
