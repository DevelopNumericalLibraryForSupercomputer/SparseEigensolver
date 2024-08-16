from array cimport array, one, two, three, four, five, six
from Comm cimport Comm, MKLComm
from Contiguous1DMap cimport Contiguous1DMap

cdef extern from "SparseTensor.hpp" namespace "SE":
    cdef cppclass SparseTensor[dimension, datatype, maptype, device]:
        pass

    cdef cppclass MKL_SparseTensor1D "SE::SparseTensor<1, double, SE::Contiguous1DMap<1>, SE::DEVICETYPE::MKL>":
        pass

    cdef cppclass MKL_SparseTensor2D "SE::SparseTensor<2, double, SE::Contiguous1DMap<2>, SE::DEVICETYPE::MKL>":
        pass
