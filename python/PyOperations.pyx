#distutils: language = c++
#PyOperations.pyx

cimport numpy as np
import numpy as np
from cpython cimport PyObject, Py_INCREF, Py_DECREF
from libcpp.vector cimport vector
np.import_array()

cdef extern from "decomposition/PyOperations.hpp" namespace "SE":
    ctypedef void   (*MatrixOneVecCallback)(const double* input_vec, double* output_vec, const int size, void* user_data)
    ctypedef void   (*MatrixMultVecCallback)(const double* input_vecs, double* output_vec, const int num_vec, const int size, void* user_data)
    ctypedef double (*GetDiagElementCallback)(int index, void* user_data)
    ctypedef void   (*GetGlobalShapeCallback)(int* shape, void* user_data)

    cdef cppclass PyTensorOperations[MTYPE, DEVICETYPE]:
        pass
    cdef cppclass Serial_double_TOp "SE::PyTensorOperations<double, SE::DEVICETYPE::MKL>":
        Serial_double_TOp(MatrixOneVecCallback matrix_one_vec_callback, MatrixMultVecCallback matrix_mult_vec_callback, GetDiagElementCallback get_diag_element_callback, GetGlobalShapeCallback get_global_shape_callback, void* user_data)

#행렬과 벡터를 곱하는 callback 함수
cdef void matonevec_wrapper(const double* input_vec, double* output_vec, const int size, void* user_data):
    cdef PyObject* py_operations = <PyObject*>user_data
    cdef object py_callback = <object>py_operations
    
    #입력 벡터를 numpy array로 변환
    cdef const double[::1] input_vec_array = <const double[:size:1]>input_vec
    cdef np.ndarray[double, ndim=1] np_input_vec = np.PyArray_SimpleNewFromData(1, &size, np.NPY_DOUBLE, <void*>input_vec)

    #python callback 함수 호출
    result = py_callback.matvec(np_input_vec)
    
    #결과를 double* output_vec에 복사
    cdef np.ndarray[double, ndim=1] np_output_vec = np.PyArray_SimpleNewFromData(1, &size, np.NPY_DOUBLE, <void*>output_vec)
    np_output_vec[...] = result

#행렬과 벡터들을 곱하는 callback 함수
cdef void matmultvec_wrapper(const double* input_vecs, double* output_vec, const int num_vec, const int size, void* user_data):
    cdef PyObject* py_operations = <PyObject*>user_data
    cdef object py_callback = <object>py_operations
    
    #입력 벡터를 numpy array로 변환
    cdef const double[::1] input_vecs_array = <const double[:num_vec*size:1]>input_vecs
    cdef np.npy_intp dims[2]
    dims[0] = num_vec
    dims[1] = size
    cdef np.ndarray[double, ndim=2] np_input_vecs = np.PyArray_SimpleNewFromData(2, dims, np.NPY_DOUBLE, <void*>input_vecs)
    #python callback 함수 호출
    result = py_callback.matvec(np_input_vecs)
    #결과를 double* output_vec에 복사
    cdef np.ndarray[double, ndim=2] np_output_vec = np.PyArray_SimpleNewFromData(2, dims, np.NPY_DOUBLE, <void*>output_vec)
    np_output_vec[...] = result

#대각원소를 받아오는 callback 함수
cdef double diagelement_wrapper(int index, void* user_data):
    cdef PyObject* py_operations = <PyObject*>user_data
    cdef object py_callback = <object>py_operations
    return py_callback.diagelement(index)

#shape을 받아오는 callback 함수
cdef void globalshape_wrapper(int* shape, void* user_data):
    cdef PyObject* py_operations = <PyObject*>user_data
    cdef object py_callback = <object>py_operations
    cdef np.npy_intp return_shape[2]
    py_callback.globalshape(return_shape)
    shape[0] = return_shape[0]
    shape[1] = return_shape[1]
    

#Python class
cdef class Tensor_Operations:
    cdef Serial_double_TOp* c_obj
    
    def __cinit__(self, mov, mmv, gde, ggs, data):
        self.c_obj = new Serial_double_TOp(mov, mmv, gde, ggs, data)

    def __dealloc__(self):
        del self.c_obj

    