# decomposition.pxd
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from DenseTensor cimport MKL_DenseTensor2D
from DecomposeResult cimport DecomposeResult_double
from TensorOperations cimport TensorOperations

cdef extern from "decomposition/Decompose.hpp" namespace "SE":
    cdef unique_ptr[DecomposeResult_double] decompose_dense(MKL_DenseTensor2D& tensor, MKL_DenseTensor2D* eigvec, string method)
    
    #cdef unique_ptr[DecomposeResult[DATATYPE]] decompose(DenseTensor[2, DATATYPE, MAPTYPE, device]& tensor, DenseTensor[2, DATATYPE, MAPTYPE, device]* eigvec, string method)
    #cdef unique_ptr[DecomposeResult[DATATYPE]] decompose(SparseTensor[2, DATATYPE, MAPTYPE, device]& tensor, DenseTensor[2, DATATYPE, MAPTYPE, device]* eigvec, string method)
    #cdef unique_ptr[DecomposeResult[DATATYPE]] decompose(TensorOperations* operations, DenseTensor[2, DATATYPE, MAPTYPE, device]* eigvec, string method)
