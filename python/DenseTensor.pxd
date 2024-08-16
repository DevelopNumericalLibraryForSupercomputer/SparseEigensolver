from array cimport array, one, two, three, four, five, six
from Comm cimport Comm, MKLComm
from Contiguous1DMap cimport Contiguous1DMap

cdef extern from "DenseTensor.hpp" namespace "SE":
    cdef cppclass DenseTensor[dimension, datatype, maptype, device]:
        pass

    cdef cppclass MKL_DenseTensor1D "SE::DenseTensor<1,double, SE::Contiguous1DMap<1>, SE::DEVICETYPE::MKL>":
        double* data
        pass

    cdef cppclass MKL_DenseTensor2D "SE::DenseTensor<2,double, SE::Contiguous1DMap<2>, SE::DEVICETYPE::MKL>":
        double* data
        pass
