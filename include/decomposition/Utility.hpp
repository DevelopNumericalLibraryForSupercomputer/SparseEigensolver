#pragma once
#include <memory>
#include <complex>

#include "../Device.hpp"
#include "../ContiguousMap.hpp"
#include "DecomposeOption.hpp"
namespace SE{

template <typename datatype>
class DecomposeResult{
public:
    DecomposeResult(const size_t num_eig, 
                    std::unique_ptr< datatype[] > real_eigvals,
                    std::unique_ptr< datatype[] > imag_eigvals
                   ): num_eig(num_eig),
                     real_eigvals(std::move(real_eigvals )),
                     imag_eigvals(std::move(imag_eigvals ))
                   {};
    const size_t num_eig=0;
    std::unique_ptr<datatype[] > real_eigvals;
    std::unique_ptr<datatype[] > imag_eigvals;
};


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
