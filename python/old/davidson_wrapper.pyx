# davidson_wrapper.pyx

cimport cython
#from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
import numpy as np


cdef extern from "../include/DenseTensor.hpp" namespace "SE":
    cdef cppclass DenseTensor[size_t, DATATYPE, MAPTYPE, DEVICETYPE]:
        pass

cdef extern from "../include/Contiguous1DMap.hpp" namespace "SE":
    cdef cppclass Contiguous1DMap[size_t dimension]:
        pass

    cdef cppclass Contiguous1DMap2D:
        cdef Contiguous1DMap[2]* thisptr
   
        def __cint__(self, list global_shape, size_t my_rank, size_t world_size, list ranks_per_dim):
            cdef std.array[size_t, len(global_shape)] global_shape_array
            cdef std.array[size_t, len(global_shape)] ranks_per_dim_array
            for a in len(global_shape):
                global_shape_array[a] = global_shape[a]
                ranks_per_dim_array[a] = ranks_per_dim[a]
            self.thisptr = new Contiguous1DMap[2](global_shape_array, my_rank, world_size, ranks_per_dim)
    
        def __dealloc__(self):
            del self.thisptr

cdef extern from "../include/decomposition/IterativeSolvers.hpp" namespace "SE":
    cdef cppclass DecomposeResult[DATATYPE]:
        pass

    cdef cppclass TensorOperations:
        pass

    cdef extern unique_ptr[DecomposeResult[DATATYPE]] davidson[DATATYPE, MAPTYPE, DEVICETYPE](TensorOperations*, DenseTensor[size_t, DATATYPE, MAPTYPE, DEVICETYPE]*) except +

#cdef class PyTensorOperations(TensorOperations):
    # Python에서 구현된 TensorOperations 메서드를 여기에 정의


#https://stackoverflow.com/questions/72931336/is-there-a-preferred-way-to-interface-with-c-stdarray-types-in-cython
#cdef extern from "<array>" namespace "std" nogil:
#    cdef cppclass arrayi2 "std::array<size_t, 2>":
#        arrayi2() except +
#        size_t& operator[](size_t)

#cdef arrayi2ToNumpy(arrayi2 arr):
#    cdef size_t[::1] view = <size_t[:2]>(&arr[0])
#    return np.asarray(view.copy())

#cdef arrayi2 numpyToArrayi2(nparr):
#    nparr = np.asarray(nparr, dtype=np.ulonglong)
#    cdef size_t[:] view = memoryview(nparr)
#    cdef arrayi2 *arr = <arrayi2 *>(&view[0])
#    return dereference(arr)

#cdef class DenseTensorCont2Ddouble:
#    cdef DenseTensor[2, DATATYPE, Contiguous1DMap[2], DEVICETYPE.MKL]* thisptr

#    def __cinit__(self, np.ndarray data):
#        # 여기서 Comm과 Map 객체를 생성하거나 기본값으로 설정해야 합니다.
#        cdef Comm[DEVICETYPE.MKL] comm = Comm[DEVICETYPE.MKL](0,1)
#        global_shape = [data.shape[0], data.shape[1]]
#        ranks_per_dim = [1,1]
#        cdef Contiguous1DMap2D map = Contiguous1DMap2D(global_shape, 0,1,ranks_per_dim)
#        self.thisptr = new DenseTensor[2, double, Contiguous1DMap[2], DEVICETYPE.MKL](comm, map.thisptr, <double*>&data[0, 0])
#
#    def __dealloc__(self):
#        del self.thisptr

#def run_davidson(PyTensorOperations operations, np.ndarray eigvec):
#    cdef DenseTensor2D[double, Contiguous1DMap, DEVICETYPE.MKL] eigvec_tensor = DenseTensor2D[double, Contiguous1DMap, DEVICETYPE.MKL](eigvec)
#    result = davidson[double, Contiguous1DMap, DEVICETYPE.MKL](&operations, &eigvec_tensor.thisptr)
#    return result