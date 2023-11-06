#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "Tensor.hpp"

#include "decomposition/Utility.hpp"
#include "decomposition/DecomposeOption.hpp"
namespace SE{

//COO sparse matrix
template<typename datatype, size_t dimension, typename computEnv, typename maptype>
class SparseTensor: public Tensor<datatype, dimension, computEnv, maptype>{
public:
    //vector of (index array, data)
    std::vector<std::pair<std::array<size_t, dimension>, datatype> > data;

    SparseTensor(){filled = false;};
    SparseTensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape);
    SparseTensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, size_t data_size);
    SparseTensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, std::vector<std::pair<std::array<size_t, dimension>, datatype> > data);

    datatype& operator()(const std::array<size_t, dimension> index);
    SparseTensor<datatype, dimension, computEnv, maptype>& operator=(const SparseTensor<datatype, dimension, computEnv, maptype> &tensor);

    void complete();
    bool get_filled(){return filled;};
    void insert_value(std::array<size_t, dimension> index, datatype value);
    void print_tensor();

    SparseTensor<datatype, dimension, computEnv, maptype> clone() {return SparseTensor<datatype, dimension, computEnv, maptype> (this->shape, this->data); };
    /*
    std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > decompose(const std::string method);

    std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > davidson();
    void preconditioner(DecomposeOption option, double* sub_eigval, double* residual, size_t block_size, double* guess);
    */
    void export_csr( const size_t dim, 
                     std::vector<size_t>& Bp,      // ROW_INDEX
                     std::vector<size_t>& Bj,      // COL_INDEX
                     std::vector<datatype>& Bx     // values
                   );
   

protected:
    bool filled = false;
private:
    size_t calculate_column( std::array<size_t, dimension> index, size_t dim);
};

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
SparseTensor<datatype, dimension, computEnv, maptype>::SparseTensor(Comm<computEnv> *_comm, maptype *_map, std::array<size_t, dimension> _shape)
: Tensor<datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    this->filled = false;
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
SparseTensor<datatype, dimension, computEnv, maptype>::SparseTensor(Comm<computEnv> *_comm, maptype *_map, std::array<size_t, dimension> _shape, size_t data_size)
: SparseTensor(_comm, _map, _shape){
    this->data.reserve(data_size);
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
SparseTensor<datatype, dimension, computEnv, maptype>::SparseTensor(Comm<computEnv> *_comm, maptype *_map, std::array<size_t, dimension> _shape, std::vector<std::pair<std::array<size_t, dimension>, datatype>> data)
: Tensor<datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    this->data = data;
    this->filled = true;
    this->complete();
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
datatype &SparseTensor<datatype, dimension, computEnv, maptype>::operator()(const std::array<size_t, dimension> index)
{
    for (size_t i = 0; i < this->data.size(); i++){
        // array equal, c++20
        //std::cout << i << " " << data[i].first[0] << " " << data[i].first[1] << " " << data[i].first[2] << " " << data[i].second << std::endl;
        if(data[i].first == index){
            return this->data[i].second;
        }
    }
    datatype* null_data;
    return *null_data;
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
inline void SparseTensor<datatype, dimension, computEnv, maptype>::insert_value(std::array<size_t, dimension> index, datatype value){
    assert(this->filled == false);
    this->data.push_back(std::make_pair(index, value));
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void SparseTensor<datatype, dimension, computEnv, maptype>::print_tensor(){
    std::cout << "print is not implemented yet." << std::endl;
    exit(-1);
}

template <>
void SparseTensor<double, 2, MKL, ContiguousMap<2> >::print_tensor(){
    std::cout << "=======================" << std::endl;
    for(auto const &i: this->data){
        for(int j=0;j<2;j++) std::cout << i.first[j] << '\t';
        std::cout << std::setw(6) << i.second << std::endl;
    }
    std::cout << "=======================" << std::endl;
    return;
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void SparseTensor<datatype, dimension, computEnv, maptype>::complete(){
    std::cout << "Sparse tensor for parallel environment is not implented" << std::endl;
    exit(1);
}

template <>
void SparseTensor<double, 2, MKL, ContiguousMap<2> >::complete()
{
    if(!this->filled && this->data.size() !=0){
        std::sort(this->data.begin(), this->data.end());
        
        for (size_t i = 0; i < data.size() - 1; i++){
            if(data[i].first == data[i+1].first){
                data[i].second += data[i+1].second;
                data.erase(std::begin(data)+i+1);
                
                i -= 1;
            }
        }
    }
    this->filled = true;
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void SparseTensor<datatype, dimension, computEnv, maptype>::export_csr(const size_t dim, std::vector<size_t> &Bp, std::vector<size_t> &Bj, std::vector<datatype> &Bx){
    // n_row : number of the dim-st indicies
    // n_col : number of rest of index combinations
    const size_t n_row = this->shape[dim];
    const size_t n_col = this->shape_mult[dimension] / n_row;
    const size_t nnz   = this->data.size();

    //compute number of non-zero entries per row of A 
    Bp.resize(n_row+1, 0);
    Bj.resize(nnz);
    Bx.resize(nnz);

    for (size_t n = 0; n < nnz; n++){
        Bp[ this->data[n].first[dim] ]+=1;
    }

    //cumsum the nnz per row to get Bp[]
    for(size_t i = 0, cumsum = 0; i < n_row; i++){
        const size_t temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz; 

    // row major
    //write Aj,Ax into Bj,Bx
    for(size_t n = 0; n < nnz; n++){
        size_t row  = this->data[n].first[dim];
        size_t dest = Bp[row];

        Bj[dest] = calculate_column(this->data[n].first, dim);
        Bx[dest] = this->data[n].second;

        Bp[row]++;
    }
    Bp.insert(Bp.begin(), 0);
    Bp.pop_back();
    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
size_t SparseTensor<datatype, dimension, computEnv, maptype>::calculate_column(std::array<size_t, dimension> index, size_t dim){
    size_t return_val = 0;
    size_t stride = 1;
    for (size_t i = 0; i < dimension; i++){
        const size_t curr_dim = dimension - i - 1;
        if(curr_dim == dim){
            continue;
        }
        return_val += stride * index[curr_dim];
        stride *= this->shape[curr_dim];
    }
    return return_val;
}
/*
template <typename datatype, size_t dimension, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > SparseTensor<datatype, dimension, computEnv, maptype>::decompose(const std::string method){
    if(method.compare("Davidson")==0){
        return davidson();
    }
    else{    
        std::cout << method << " is not implemented yet." << std::endl;
        exit(-1);
    }
    
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > SparseTensor<datatype, dimension, computEnv, maptype>::davidson(){
    std::cout << "davidson is not implemented yet." << std::endl;
    exit(-1);
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void SparseTensor<datatype, dimension, computEnv, maptype>::preconditioner(DecomposeOption option, double* sub_eigval, double* residual, size_t block_size, double* guess){
    std::cout << "invalid preconditioner." << std::endl;
    exit(-1);
}
*/

/*
template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void SparseTensor<datatype, dimension, computEnv, maptype>::read_csr(const int *row_ptr, const int *col_ind, const datatype *val, const size_t row_size,
                                                            const size_t col_size, const size_t val_size, const std::string order, bool pass_complete)
{
    /* clear all existing information and load value from csr matrix 
     * Input)
     * row_ptr: The row_ptr vector stores the locations in the val vector that start a row
     * col_ind: The col_ind vector stores the column index 
     * val    : non-zero values of matrix elements
     * order  : order of the input tensor's index. "C" or "F"
     * Output)
     * 
    

}
*/
}
