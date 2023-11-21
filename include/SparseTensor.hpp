#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "Tensor.hpp"

#include "decomposition/DecomposeOption.hpp"
namespace SE{

//COO sparse matrix
// partition : row-wise
// a00 a01 a02
// a10 a11 a12
// --> a00 a10 a01 a11 a02 a12
// --> a00 a01 a02 / a10 a11 a12
template<typename datatype, size_t dimension, typename computEnv, typename maptype>
class Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>{
private:
    bool filled = false;
    size_t calculate_column( std::array<size_t, dimension> index, size_t dim);
    void complete_local();

public:
    const STORETYPE store_type = STORETYPE::COO;
    std::array<size_t, dimension> shape;
    std::array<size_t, dimension+1> shape_mult;

    Comm<computEnv>* comm;
    maptype* map;
    //vector of (index array, data)
    std::vector<std::pair<std::array<size_t, dimension>, datatype> > data;

    Tensor(){filled = false;};
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape);
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, size_t data_size);
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, std::vector<std::pair<std::array<size_t, dimension>, datatype> > data);

    datatype& operator()(const std::array<size_t, dimension> index);
    Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>& operator=(const Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype> &tensor);

    void complete();
    bool get_filled(){return filled;};
    void insert_value(std::array<size_t, dimension> index, datatype value);
    void print_tensor();

    Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype> clone() {
        return Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>(this->comm, this->map, this->shape, this->data);
    };
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

};

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::Tensor(Comm<computEnv> *_comm, maptype* _map, std::array<size_t, dimension> _shape): comm(_comm), shape(_shape){
    this->map = _map;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    this->filled = false;
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::Tensor(Comm<computEnv> *_comm, maptype* _map, std::array<size_t, dimension> _shape, size_t data_size)
: Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    this->data.reserve(data_size);
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::Tensor(Comm<computEnv> *_comm, maptype* _map, std::array<size_t, dimension> _shape, std::vector<std::pair<std::array<size_t, dimension>, datatype>> data)
: Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    this->data = data;
    this->filled = true;
    this->complete();
}

template <>
Tensor<STORETYPE::COO, double, 2, MPI, ContiguousMap<2> >::Tensor(Comm<MPI> *_comm, ContiguousMap<2>* _map, std::array<size_t, 2> _shape, std::vector<std::pair<std::array<size_t, 2>, double>> data)
: Tensor<STORETYPE::COO, double, 2, MPI, ContiguousMap<2> >(_comm, _map, _shape){
    this->data.reserve(data.size());
    for(auto iter = data.begin(); iter != data.end(); iter++){
        int this_rank = _map->get_my_rank_from_global_index(iter->first[0], 0);
        if(_comm->rank == this_rank) this->data.emplace_back(iter->first, iter->second);
    }
    this->filled = true;
    this->complete();
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
datatype &Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::operator()(const std::array<size_t, dimension> index){
    for (size_t i = 0; i < this->data.size(); i++){
        // array equal, c++20
        //std::cout << i << " " << data[i].first[0] << " " << data[i].first[1] << " " << data[i].first[2] << " " << data[i].second << std::endl;
        if(data[i].first == index){
            return this->data[i].second;
        }
    }
    std::cout << "no element!" << std::endl;
    exit(-1);
    this->data.push_back(std::make_pair(index, 0.0));
    return this->data.back().second;
}

template <>
double &Tensor<STORETYPE::COO, double, 2, MPI, ContiguousMap<2> >::operator()(const std::array<size_t, 2> index){
    std::array<size_t, 2> local_index = this->map->get_local_array_index(index, 1, this->comm->rank); //assert that given index is in this thread.
    for (size_t i = 0; i < this->data.size(); i++){
        if(data[i].first == index){
            return this->data[i].second;
        }
    }
    this->data.push_back(std::make_pair(index, 0.0));
    return this->data.back().second;
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::insert_value(std::array<size_t, dimension> index, datatype value){
    //std::cout << "inside default insert_val" << std::endl;
    assert(this->filled == false);
    //this->data.push_back(std::make_pair(index, value));
    this->data.emplace_back(index, value);
}


template <>
void Tensor<STORETYPE::COO, double, 2, MPI, ContiguousMap<2> >::insert_value(std::array<size_t, 2> index, double value){
    //std::cout << "inside D2MC<2> insert_val" << std::endl;
    //matrix는 col major로 저장됨
    //공간상에서 자르는 index 는 row index로 자름.
    // a00 a01 a02
    // a10 a11 a12
    // --> a00 a10 a01 a11 a02 a12
    // --> a00 a01 a02 / a10 a11 a12
    int my_rank = comm->rank;
    size_t target_rank = map->get_my_rank_from_global_index(index[0], 0);
    //std::cout << "insertval, myrank : " << my_rank << " target_rank : " << target_rank << " val : " << index[0] << " " << index[1] << " " << value << std::endl;
    if(my_rank == target_rank){
        this->data.emplace_back(index, value);
    }
    else{
        std::cout << "( " << index[0] << ", " << index[1] << " ) cannot be inserted in the process number " << my_rank << std::endl;
    }
    /*
    size_t row, col;
    double recv_value;
    else{
        row = index[0];
        col = index[1];
        if(comm->rank = my_rank){
            send<int>(&row, 1, target_rank);
            send<int>(&col, 1, target_rank);
            send<double>(&value, 1, target_rank);
        }
        else if(comm->rank == target_rank){
            recv<int>(&row, 1, my_rank);
            recv<int>(&col, 1, my_rank);
            recv<double>(&value, 1, my_rank);
        }
        comm->barrier();
    }
    */
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::print_tensor(){
    std::cout << "print is not implemented yet." << std::endl;
    exit(-1);
}

template <>
void Tensor<STORETYPE::COO, double, 2, MKL, ContiguousMap<2> >::print_tensor(){
    std::cout << "=======================" << std::endl;
    for(auto const &i: this->data){
        for(int j=0;j<2;j++) std::cout << i.first[j] << '\t';
        std::cout << std::setw(6) << i.second << std::endl;
    }
    std::cout << "=======================" << std::endl;
    return;
}

template <>
void Tensor<STORETYPE::COO, double, 2, MPI, ContiguousMap<2> >::print_tensor(){
    std::cout << "=======================" << std::endl;
    for(auto const &i: this->data){
        for(int j=0;j<2;j++) std::cout << i.first[j] << '\t';
        std::cout << std::setw(6) << i.second << std::endl;
    }
    std::cout << "=======================" << std::endl;
    return;
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::complete_local(){
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

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::complete(){
    complete_local();
}
/*
template <>
void SparseTensor<double, 2, MPI, ContiguousMap<2> >::complete()
{
    
    //get total tensor size and store at rank==0
    size_t local_size = this->size();
    size_t total_size = 0.0;
    comm->reduce<size_t>(&local_size, 1, &total_size, SEop::SUM, 0);

    //std::pair<std::array<size_t, 2>, double>* whole_data;
    size_t* rows, columns;
    double* values;
    if(this->comm->rank==0){
        rows = malloc<size_t>(total_size);
        columns = malloc<size_t>(total_size);
        values = malloc<double>(total_size);
    }
    comm->gather(this->data(), local_size, )
    // Gather elements from all processes
    // MPI_Gather(...);

    // Process 0 gathers all elements and performs sorting
    if (this->comm->rank == 0) {
        // Sort elements based on row indices
        std::sort(this->data.begin(), this->data.end(), [](const auto& a, const auto& b) {
            return a.first[0] < b.first[0];
        });

        // Distribute elements to the corresponding processes
        // MPI_Scatter(...);
    }

    // Update local_elements with received_elements
    
    complete_local();
}
*/

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::export_csr(const size_t dim, std::vector<size_t> &Bp, std::vector<size_t> &Bj, std::vector<datatype> &Bx){
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

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
size_t Tensor<STORETYPE::COO, datatype, dimension, computEnv, maptype>::calculate_column(std::array<size_t, dimension> index, size_t dim){
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
