# distutils: language = c++
from libcpp.memory cimport unique_ptr

from array cimport array, one, two, three, four, five, six
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "Map.hpp" namespace "SE":
    cdef cppclass Map[DIMENSION, MTYPE]:
        pass
        #Comm() except+
        #Comm(size_t rank, size_t world_size) except+
        #Comm[DEVICETYPE]* clone()

cdef extern from "Contiguous1DMap.hpp" namespace "SE":
    cdef cppclass Contiguous1DMap[DIMENSION]:
        Contiguous1DMap() except+
        Contiguous1DMap(const array[size_t, DIMENSION] global_shape, const size_t my_rank, const size_t world_size) except+
        Contiguous1DMap(const array[size_t, DIMENSION] global_shape, const size_t my_rank, const size_t world_size, const array[bool, DIMENSION] is_parallel ) except+
        Contiguous1DMap(const array[size_t, DIMENSION] global_shape, const size_t my_rank, const size_t world_size, const array[size_t, DIMENSION] ranks_per_dim ) except+

        Contiguous1DMap[DIMENSION]* clone()
    
        size_t get_num_local_elements()
    
        # global index <-> local index
        size_t local_to_global(size_t local_index) 
        size_t global_to_local(size_t global_index) 
        array[size_t, DIMENSION] local_to_global(array[size_t, DIMENSION] local_array_index) 
        array[size_t, DIMENSION] global_to_local(array[size_t, DIMENSION] global_array_index) 
                                                                                
        # local array index <-> local index 
        size_t  unpack_local_array_index(array[size_t, DIMENSION] local_array_index) 
        array[size_t, DIMENSION] pack_local_index(size_t local_index)
        size_t  unpack_global_array_index(array[size_t, DIMENSION] global_array_index) 
        array[size_t, DIMENSION] pack_global_index(size_t global_index) 

        size_t find_rank_from_global_index(size_t global_index) 
        size_t find_rank_from_global_array_index(array[size_t, DIMENSION] global_array_index) 

        #vector[array[size_t, DIMENSION]] get_all_local_shape() 
        size_t get_split_dim() 

        vector[array[size_t, DIMENSION]] all_local_shape
        size_t split_dim
        initialize()

    