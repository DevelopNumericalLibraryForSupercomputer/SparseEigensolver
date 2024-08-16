from array cimport array, one, two, three, four, five, six
from Contiguous1DMap cimport Contiguous1DMap
from Comm cimport Comm, MKLComm
from DenseTensor cimport DenseTensor, DenseTensor1D, DenseTensor2D
from Type cimport DEVICETYPE
from libcpp cimport bool
from DecomposeResult cimport DecomposeResult

cimport numpy as np


# python version의 1D, 2D tensor를 만들기
cdef make_2D_tensor(np.ndarray[double,ndim=2] data):
    cdef array[int, two] global_shape
    cdef array[bool, two] is_parallel
    is_parallel[0] = True
    is_parallel[1] = False
    global_shape[0] = data.shape[0]
    global_shape[1] = data.shape[1]
    cdef MKLComm comm = MKLComm(0,1)
    cdef int my_rank = comm.get_rank()
    cdef int global_size = comm.get_world_size()
    cdef Contiguous1DMap[two] map = Contiguous1DMap[two](global_shape, my_rank, global_size, is_parallel)
    cdef DenseTensor2D* tensor = new DenseTensor2D(comm, map, &data[0,0])
    
    cdef DecomposeResult[double] test
    
    
def test(np.ndarray[double,ndim=2] data):
    return make_2D_tensor(data)
