# cython: language_level=3
# distutils: language = c++

from array cimport array, one, two, three, four, five, six
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "../include/Contiguous1DMap.hpp" namespace "SE":
    cppclass Contiguous1DMap[dimension]:
        Contiguous1DMap() except+
        Contiguous1DMap(const array[size_t, dimension] global_shape, const size_t my_rank, const size_t world_size) except+
        Contiguous1DMap(const array[size_t, dimension] global_shape, const size_t my_rank, const size_t world_size, const array[bool, dimension] is_parallel ) except+
        Contiguous1DMap(const array[size_t, dimension] global_shape, const size_t my_rank, const size_t world_size, const array[size_t, dimension] ranks_per_dim ) except+

        Contiguous1DMap[dimension]* clone()
    
        size_t get_num_local_elements()
    
        # global index <-> local index
        size_t local_to_global(size_t local_index) 
        size_t global_to_local(size_t global_index) 
        array[size_t, dimension] local_to_global(array[size_t, dimension] local_array_index) 
        array[size_t, dimension] global_to_local(array[size_t, dimension] global_array_index) 
                                                                                
        # local array index <-> local index 
        size_t  unpack_local_array_index(array[size_t, dimension] local_array_index) 
        array[size_t, dimension] pack_local_index(size_t local_index)
        size_t  unpack_global_array_index(array[size_t, dimension] global_array_index) 
        array[size_t, dimension] pack_global_index(size_t global_index) 

        size_t find_rank_from_global_index(size_t global_index) 
        size_t find_rank_from_global_array_index(array[size_t, dimension] global_array_index) 

        #vector[array[size_t, dimension]] get_all_local_shape() 
        size_t get_split_dim() 

        vector[array[size_t, dimension]] all_local_shape
        size_t split_dim
        initialize()