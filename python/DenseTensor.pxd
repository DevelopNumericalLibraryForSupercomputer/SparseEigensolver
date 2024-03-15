from array cimport array, one, two, three, four, five, six
from Comm cimport Comm, MKLComm
from Contiguous1DMap cimport Contiguous1DMap

cdef extern from "../include/DenseTensor.hpp" namespace "SE":
    cdef cppclass DenseTensor[dimension, datatype, maptype, device]:
        DenseTensor(Comm[device]& comm, Contiguous1DMap[dimension]& map) except+
        DenseTensor(Comm[device]& comm, Contiguous1DMap[dimension]& map, double* data) except+
        DenseTensor(DenseTensor& tensor) except+

        Comm[device] comm
        Contiguous1DMap[dimension] map
        double* data

        void global_insert_value(array[size_t, one] global_array_index, double value)
        void local_insert_value(array[size_t, one] local_array_index, double value)
        void global_insert_value(size_t global_index, double value)
        void local_insert_value(size_t local_index, double value)

        void global_set_value(array[size_t, one] global_array_index, double value)
        void local_set_value(array[size_t, one] local_array_index, double value)
        void global_set_value(size_t global_index, double value)
        void local_set_value(size_t local_index, double value)

    cdef cppclass DenseTensor1D "SE::DenseTensor<1,double, SE::Contiguous1DMap<1>, SE::DEVICETYPE::MKL>":
        DenseTensor1D() except+
        DenseTensor1D(MKLComm& comm, Contiguous1DMap[one]& map) except+
        DenseTensor1D(MKLComm& comm, Contiguous1DMap[one]& map, double* data) except+
        DenseTensor1D(DenseTensor1D& tensor) except+

        MKLComm comm
        Contiguous1DMap[one] map
        double* data

        void global_insert_value(array[size_t, one] global_array_index, double value)
        void local_insert_value(array[size_t, one] local_array_index, double value)
        void global_insert_value(size_t global_index, double value)
        void local_insert_value(size_t local_index, double value)

        void global_set_value(array[size_t, one] global_array_index, double value)
        void local_set_value(array[size_t, one] local_array_index, double value)
        void global_set_value(size_t global_index, double value)
        void local_set_value(size_t local_index, double value)

    cdef cppclass DenseTensor2D "SE::DenseTensor<2,double, SE::Contiguous1DMap<2>, SE::DEVICETYPE::MKL>":
        DenseTensor2D() except+
        DenseTensor2D(MKLComm& comm, Contiguous1DMap[two]& map) except+
        DenseTensor2D(MKLComm& comm, Contiguous1DMap[two]& map, double* data) except+
        DenseTensor2D(DenseTensor2D& tensor) except+

        MKLComm comm
        Contiguous1DMap[two] map
        double* data

        void global_insert_value(array[size_t, two] global_array_index, double value)
        void local_insert_value(array[size_t, two] local_array_index, double value)
        void global_insert_value(size_t global_index, double value)
        void local_insert_value(size_t local_index, double value)
        
        void global_set_value(array[size_t, two] global_array_index, double value)
        void local_set_value(array[size_t, two] local_array_index, double value)
        void global_set_value(size_t global_index, double value)
        void local_set_value(size_t local_index, double value)