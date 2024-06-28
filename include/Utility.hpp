#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <array>
#include <vector>

namespace SE{
template <size_t dimension>
void cumprod(const std::array<size_t, dimension>& shape, std::array<size_t, dimension+1>& shape_mult, std::string indexing="F"){
    /* Ex1)
     * shape = {2, 3, 4}, indexing="F"
     * shape_mult = {1, 2, 6, 24}
     * Ex2)
     * shape = {2, 3, 4}, indexing="C"
     * shape_mult = {1, 4, 12, 24}
     */
    shape_mult[0] = 1;
    if (indexing == "F"){
        for (size_t i = 0; i < dimension; ++i) {
            shape_mult[i+1] = shape_mult[i] * shape[i];
        }
    }
    else if(indexing == "C"){
        for (size_t i = 0; i < dimension; ++i) {
            shape_mult[i+1] = shape_mult[i] * shape[dimension-i-1];
        }
    }
}

template<typename DATATYPE>
std::vector<size_t> sort_indicies(const DATATYPE* data_array, const size_t array_size){
    std::vector<size_t> idx;
    idx.resize(array_size);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::stable_sort(std::begin(idx), std::end(idx), [data_array](size_t i1, size_t i2) {return data_array[i1] < data_array[i2];});
    return idx;
}

/* Auxiliary routine: printing eigenvalues */
void print_eigenvalues( const std::string desc, size_t n, double* wr, double* wi ) {
   std::cout << "\n" << desc << std::endl;
   for(size_t j = 0; j < n; j++ ) {
      if( wi[j] == (double)0.0 ) {
         printf( " %6.8f", wr[j] );
      } else {
         printf( " (%6.2f,%6.2f)", wr[j], wi[j] );
      }
   }
   std::cout << std::endl;
}

/* Auxiliary routine: printing eigenvectors */
void print_eigenvectors( const std::string desc, size_t n, double* wi, double* v, size_t ldv ) {
   size_t i, j;
   std::cout << "\n" << desc << std::endl;
   for( i = 0; i < n; i++ ) {
      j = 0;
      while( j < n ) {
         if( wi[j] == (double)0.0 ) {
            printf( " %6.8f", v[i+j*ldv] );
            j++;
         } else {
            printf( " (%6.2f,%6.2f)", v[i+j*ldv], v[i+(j+1)*ldv] );
            printf( " (%6.2f,%6.2f)", v[i+j*ldv], -v[i+(j+1)*ldv] );
            j += 2;
         }
      }
      std::cout << std::endl;
   }
}

}
