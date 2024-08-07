#pragma once
#include <iostream>
#include "Tensor.hpp"

//#include "decomposition/DecomposeOption.hpp"
namespace SE{

//COO sparse matrix
// partition : row-wise
// a00 a01 a02
// a10 a11 a12
// --> a00 a10 a01 a11 a02 a12
// --> a00 a01 a02 / a10 a11 a12
template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
class Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>{
private:
    bool filled = false;
    int calculate_column( std::array<int, dimension> index, int dim);
    void complete_local();

public:
    const STORETYPE store_type = STORETYPE::COO;
    std::array<int, dimension> shape;
    std::array<int, dimension+1> shape_mult;

    const Comm<device>* comm;
    const MAPTYPE map;
    //vector of (index array, data)

    Tensor(){filled = false;};
    //Tensor(Comm<device>* _comm, MAPTYPE* _map, std::array<int, dimension> _shape);
    //Tensor(Comm<device>* _comm, MAPTYPE* _map, std::array<int, dimension> _shape, int data_size);
    //Tensor(Comm<device>* _comm, MAPTYPE* _map, std::array<int, dimension> _shape, std::vector<std::pair<std::array<int, dimension>, DATATYPE> > data);


    DATATYPE& operator()(const std::array<int, dimension> index);
    Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>& operator=(const Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE> &tensor);


    void export_csr( const int dim, 
                     std::vector<int>& Bp,      // ROW_INDEX
                     std::vector<int>& Bj,      // COL_INDEX
                     std::vector<DATATYPE>& Bx     // values
                   );

};

template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::Tensor(const Comm<device>* _comm,
                                                                        const std::array<int, dimension> _shape,
                                                                        const bool is_sliced,
                                                                        const int sliced_dimension)
: comm(_comm), shape(_shape), map( is_sliced ? MAPTYPE(_shape, _comm->get_world_size(), sliced_dimension) : MAPTYPE(_shape, _comm->get_world_size()) ){
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    this->filled = false;
};

template <typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::Tensor(const Comm<device> *_comm, const std::array<int, dimension> _shape, const int data_size, const bool is_sliced, const int sliced_dimension)
            : Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>(_comm, _shape, is_sliced, sliced_dimension){
    this->data.reserve(data_size);
};

template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::Tensor(const Comm<device>* _comm, const std::array<int, dimension> _shape, const std::vector<std::pair<std::array<int, dimension>, DATATYPE>> data, const bool is_sliced, const int sliced_dimension)
            : Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>(_comm, _shape, is_sliced, sliced_dimension){
    this->data = data;
    this->filled = true;
    this->complete();
}

template <>
Tensor<STORETYPE::COO, double, 2, SEMpi, ContiguousMap<2> >::Tensor(const Comm<SEMpi>* _comm, const std::array<int, 2> _shape, const std::vector<std::pair<std::array<int, 2>, double>> data, const bool is_sliced, const int sliced_dimension)
: Tensor<STORETYPE::COO, double, 2, SEMpi, ContiguousMap<2> >(_comm, _shape, is_sliced, sliced_dimension){
    this->data.reserve(data.size());
    for(auto iter = data.begin(); iter != data.end(); iter++){
        if(this->map.is_sliced){
            int this_rank = this->map.get_my_rank_from_global_index(iter->first);
            if(_comm->get_rank() == this_rank) this->data.emplace_back(iter->first, iter->second);
        }
        else{
            this->data.emplace_back(iter->first, iter->second);
        }
    }
    this->filled = true;
    this->complete();
}

template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
DATATYPE &Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::operator()(const std::array<int, dimension> index){
    std::array<int, 2> local_index = this->map.get_local_array_index(index, this->comm->get_rank()); //assert that given index is in this thread.
    for (int i = 0; i < this->data.size(); i++){
        if(data[i].first == index){
            // array equal, c++20
            //std::cout << i << " " << data[i].first[0] << " " << data[i].first[1] << " " << data[i].first[2] << " " << data[i].second << std::endl;
            return this->data[i].second;
        }
    }
    this->data.push_back(std::make_pair(index, 0.0));
    return this->data.back().second;
}

template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
void Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::insert_value(std::array<int, dimension> index, DATATYPE value){
    //matrix는 col major로 저장됨
    //공간상에서 자르는 index 는 row index로 자름.
    // a00 a01 a02
    // a10 a11 a12
    // --> a00 a10 a01 a11 a02 a12
    // --> a00 a01 a02 / a10 a11 a12
    int my_rank = comm->get_rank();
    int target_rank = 0;
    if(this->map.is_sliced){
        target_rank = this->map.get_my_rank_from_global_index(index);
    }

    assert(this->filled == false);
    if(my_rank == target_rank){
        this->data.emplace_back(index, value);
    }
}

/*
template <>
void Tensor<STORETYPE::COO, double, 2, SEMpi, ContiguousMap<2> >::insert_value(std::array<int, 2> index, double value){
    //std::cout << "inside D2MC<2> insert_val" << std::endl;
    //matrix는 col major로 저장됨
    //공간상에서 자르는 index 는 row index로 자름.
    // a00 a01 a02
    // a10 a11 a12
    // --> a00 a10 a01 a11 a02 a12
    // --> a00 a01 a02 / a10 a11 a12
    int my_rank = comm->get_rank();
    int target_rank = this->map.get_my_rank_from_global_index(index[0], 0);
    //std::cout << "insertval, myrank : " << my_rank << " target_rank : " << target_rank << " val : " << index[0] << " " << index[1] << " " << value << std::endl;
    if(my_rank == target_rank){
        this->data.emplace_back(index, value);
    }
    else{
        std::cout << "( " << index[0] << ", " << index[1] << " ) cannot be inserted in the process number " << my_rank << std::endl;
    }
}
*/
template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
void Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::print() const{
    std::cout << "print is not implemented yet." << std::endl;
    exit(-1);
}

template <>
void Tensor<STORETYPE::COO, double, 2, SEMkl, ContiguousMap<2> >::print() const{
    for(auto const &i: this->data){
        for(int j=0;j<2;j++) std::cout << i.first[j] << '\t';
        std::cout << std::setw(6) << i.second << std::endl;
    }
    return;
}template <>
void Tensor<STORETYPE::COO, double, 2, SEMpi, ContiguousMap<2> >::print() const{
    for(auto const &i: this->data){
        for(int j=0;j<2;j++) std::cout << i.first[j] << '\t';
        std::cout << std::setw(6) << i.second << std::endl;
    }
    return;
}

template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
void Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::print(const std::string& name) const{
    if(!this->map.is_sliced){
        if(this->comm->get_rank() == 0){
            std::cout << name << " : " << std::endl;
            print();
            std::cout << "=======================" << std::endl;
        }
    }
    else{
        std::cout << name << " : (rank " << this->comm->get_rank() << ")" << std::endl;
        print();
    }
}


template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
void Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::complete(){
    if(!this->filled && this->data.size() !=0){
        std::sort(this->data.begin(), this->data.end());
        for (int i = 0; i < data.size() - 1; i++){
            if(data[i].first == data[i+1].first){
                data[i].second += data[i+1].second;
                data.erase(std::begin(data)+i+1);
                
                i -= 1;
            }
        }
    }
    this->filled = true;
}

/*
template <>
void SparseTensor<double, 2, SEMpi, ContiguousMap<2> >::complete()
{
    
    //get total tensor size and store at rank==0
    int local_size = this->size();
    int total_size = 0.0;
    comm->reduce<int>(&local_size, 1, &total_size, SEop::SUM, 0);

    //std::pair<std::array<int, 2>, double>* whole_data;
    int* rows, columns;
    double* values;
    if(this->comm->get_rank()==0){
        rows = malloc<int>(total_size);
        columns = malloc<int>(total_size);
        values = malloc<double>(total_size);
    }
    comm->gather(this->data(), local_size, )
    // Gather elements from all processes
    // MPI_Gather(...);

    // Process 0 gathers all elements and performs sorting
    if (this->comm->get_rank() == 0) {
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

template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
void Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::export_csr(const int dim, std::vector<int> &Bp, std::vector<int> &Bj, std::vector<DATATYPE> &Bx){
    // n_row : number of the dim-st indicies
    // n_col : number of rest of index combinations
    const int n_row = this->shape[dim];
    const int n_col = this->shape_mult[dimension] / n_row;
    const int nnz   = this->data.size();

    //compute number of non-zero entries per row of A 
    Bp.resize(n_row+1, 0);
    Bj.resize(nnz);
    Bx.resize(nnz);

    for (int n = 0; n < nnz; n++){
        Bp[ this->data[n].first[dim] ]+=1;
    }

    //cumsum the nnz per row to get Bp[]
    for(int i = 0, cumsum = 0; i < n_row; i++){
        const int temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz; 

    // row major
    //write Aj,Ax into Bj,Bx
    for(int n = 0; n < nnz; n++){
        int row  = this->data[n].first[dim];
        int dest = Bp[row];

        Bj[dest] = calculate_column(this->data[n].first, dim);
        Bx[dest] = this->data[n].second;

        Bp[row]++;
    }
    Bp.insert(Bp.begin(), 0);
    Bp.pop_back();
    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

template<typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
int Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::calculate_column(std::array<int, dimension> index, int dim){
    int return_val = 0;
    int stride = 1;
    for (int i = 0; i < dimension; i++){
        const int curr_dim = dimension - i - 1;
        if(curr_dim == dim){
            continue;
        }
        return_val += stride * index[curr_dim];
        stride *= this->shape[curr_dim];
    }
    return return_val;
}
/*
template <typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE, dimension, device, MAPTYPE> > SparseTensor<DATATYPE, dimension, device, MAPTYPE>::decompose(const std::string method){
    if(method.compare("Davidson")==0){
        return davidson();
    }
    else{    
        std::cout << method << " is not implemented yet." << std::endl;
        exit(-1);
    }
    
}

template <typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE, dimension, device, MAPTYPE> > SparseTensor<DATATYPE, dimension, device, MAPTYPE>::davidson(){
    std::cout << "davidson is not implemented yet." << std::endl;
    exit(-1);
}

template <typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
void SparseTensor<DATATYPE, dimension, device, MAPTYPE>::preconditioner(DecomposeOption option, double* sub_eigval, double* residual, int block_size, double* guess){
    std::cout << "invalid preconditioner." << std::endl;
    exit(-1);
}
*/

/*
template <typename DATATYPE, int dimension, DEVICETYPE device, typename MAPTYPE>
void SparseTensor<DATATYPE, dimension, device, MAPTYPE>::read_csr(const int *row_ptr, const int *col_ind, const DATATYPE *val, const int row_size,
                                                            const int col_size, const int val_size, const std::string order, bool pass_complete)
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
