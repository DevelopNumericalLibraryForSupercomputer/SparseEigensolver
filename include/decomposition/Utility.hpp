#pragma once
#include <memory>
#include <complex>

#include "../DenseTensor.hpp"
#include "../Device.hpp"
#include "../ContiguousMap.hpp"
namespace SE{

typedef enum{//opertor for allreduce
    Davidson,
    Direct,
} DecomposeMethod;


template <typename datatype, size_t dimension, typename comm, typename map>
class DecomposeResult{
public:
    DecomposeResult(const size_t num_eig, 
                    std::unique_ptr< std::vector<datatype > > real_eigavals,
                    std::unique_ptr< std::vector<datatype > > imag_eigavals
                   ): num_eig(num_eig),
                     real_eigvals(std::move(real_eigvals) ),
                     imag_eigvals(std::move(imag_eigvals) )
                   {};
    const size_t num_eig=0;
    std::unique_ptr<std::vector<datatype > > real_eigvals;
    std::unique_ptr<std::vector<datatype > > imag_eigvals;

};


//void print_matrix(const DenseTensor<double,2,Comm<PROTOCOL::SERIAL>,ContiguousMap<2> >& M){
//    for(size_t i = 0; i < M.shape[0]; ++i){
//        for(size_t j = 0; j < M.shape[1]; ++j){
//            //std::cout << M({i,j}) << " ";
//            std::cout << M.data[i+M.shape[0]*j] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//}

/* Auxiliary routine: printing eigenvalues */
void print_eigenvalues( char* desc, MKL_INT n, double* wr, double* wi ) {
        MKL_INT j;
        printf( "\n %s\n", desc );
   for( j = 0; j < n; j++ ) {
      if( wi[j] == (double)0.0 ) {
         printf( " %6.8f", wr[j] );
      } else {
         printf( " (%6.2f,%6.2f)", wr[j], wi[j] );
      }
   }
   printf( "\n" );
}

/* Auxiliary routine: printing eigenvectors */
void print_eigenvectors( char* desc, MKL_INT n, double* wi, double* v, MKL_INT ldv ) {
        MKL_INT i, j;
        printf( "\n %s\n", desc );
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
      printf( "\n" );
   }
}

}
