#pragma once
#include <array>
#include <vector>
#include <memory>

#include "Comm.hpp"
#include "Map.hpp"
//#include "Contiguous1DMap.hpp"
//#include "decomposition/DecomposeResult.hpp"

namespace SE{


template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE DEVICETYPE, STORETYPE STORETYPE> 
class Tensor{
using array_d = std::array<int, dimension>;
using INTERNALTYPE = typename std::conditional< STORETYPE==STORETYPE::DENSE,  DATATYPE* , std::vector<std::pair<array_d, DATATYPE> > >::type; 

public:
    //const STORETYPE STORETYPE = STORETYPE;
    const std::unique_ptr<Comm<DEVICETYPE> > ptr_comm;
    const std::unique_ptr<Map<dimension,mtype>> ptr_map;
    INTERNALTYPE data; 

    // constructor
    Tensor(){
        filled=false;
    }

    Tensor(const std::unique_ptr<Comm<DEVICETYPE> >& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map):
	ptr_comm(std::unique_ptr<Comm<DEVICETYPE> >(ptr_comm->clone())),
	ptr_map(std::unique_ptr<Map<dimension,mtype>> (ptr_map->clone()) )
	{
        filled=false;
    }
    //Tensor(const std::unique_ptr<Comm<DEVICETYPE> >& comm, const std::unique_ptr<Map<dimension,mtype>>& map, int reserve_size): comm(comm), map(map) {//_internal_dataype data reserve};
    Tensor(const std::unique_ptr<Comm<DEVICETYPE> >& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map, INTERNALTYPE& data): 
	ptr_comm(std::unique_ptr<Comm<DEVICETYPE> >(ptr_comm->clone())),
	ptr_map(std::unique_ptr<Map<dimension,mtype>> (ptr_map->clone()) ),
	data(data)
	{
        filled=false;
    } 

    Tensor(const Tensor<dimension,DATATYPE,mtype,DEVICETYPE,STORETYPE>& tensor): 
	ptr_comm(std::unique_ptr<Comm<DEVICETYPE> >(ptr_comm->clone())),
	ptr_map(std::unique_ptr<Map<dimension,mtype>> (ptr_map->clone()) )
	{ 
        data = tensor.copy_data();
        filled=false;
    }

    //destructor
    ~Tensor() {
        // Depending on the storage type, perform cleanup
        if constexpr (STORETYPE == STORETYPE::DENSE) {
            // If the data is STORETYPEd as a pointer, delete it
            if (data != nullptr) {
                delete[] data;
                data = nullptr;
            }
        } else {
            // If the data is STORETYPEd as a vector, no cleanup needed
        }
    }

    // copy functions
    std::unique_ptr<Comm<DEVICETYPE> > copy_comm() const {
        return std::unique_ptr<Comm<DEVICETYPE> > (ptr_comm->clone());
    }

    std::unique_ptr<Map<dimension,mtype>> copy_map() const {
        return std::unique_ptr<Map<dimension,mtype>>(ptr_map->clone());
    }

    virtual INTERNALTYPE copy_data() const=0;

    // clone function 
    virtual std::unique_ptr<Tensor<dimension,DATATYPE,mtype,DEVICETYPE,STORETYPE> > clone(bool call_complete=false) const=0;

    // insert function (add value ) 
    virtual void global_insert_value(const array_d global_array_index, const DATATYPE value)=0;
    virtual void local_insert_value (const array_d local_array_index,  const DATATYPE value)=0;
    virtual void global_insert_value(const int global_index,        const DATATYPE value)=0;
    virtual void local_insert_value (const int local_index,         const DATATYPE value)=0;

    // set function 
    virtual void global_set_value(const array_d global_array_index, const DATATYPE value)=0;
    virtual void local_set_value (const array_d local_array_index,  const DATATYPE value)=0;
    virtual void global_set_value(const int global_index,        const DATATYPE value)=0;
    virtual void local_set_value (const int local_index,         const DATATYPE value)=0;


    // access function, it works only when filled==false
    DATATYPE operator()(const int local_index) const {
        assert(this->filled==false);
        return this->data[local_index];
    }

    // Sparse Tensor Only 
    void complete(bool reuse=true){};
    bool get_filled() const {
        return filled;
    };
    // print functions
//    void print() const;
//    void print(const std::string& name) const;

    // get functions
//    template<typename... Numbers> 
//    DATATYPE operator[] (Numbers... numbers)
//    {
//        static_assert sizeof...(numbers)==dimension;
//        return this->operator()({numbers...});
//    };

//    DATATYPE operator() (const array_d idx);
    friend std::ostream& operator<< (std::ostream& stream, const Tensor<dimension,DATATYPE,mtype,DEVICETYPE,STORETYPE>& tensor){
    //void print_tensor_info() const{
        if(tensor.ptr_comm->get_rank() == 0){
            stream << "========= Tensor Info =========" <<std::endl;
            stream << "dimension: " << dimension<< "\n" 
                   << "DATATYPE: "  << typeid(DATATYPE).name()
                   << "   shape: ("  ;
            for (auto shape_i : tensor.ptr_map->get_global_shape()){
                stream << shape_i << ",";
            }
            stream << ")\n"
                   << "Map<dimension,mtype>: "   << typeid(Map<dimension,mtype>).name() << "\n" 
                   << "DEVICETYPE: "<< int(DEVICETYPE)<<"\n"
                   << "STORETYPE: " << int(STORETYPE) <<  std::endl;   
        }
        return stream;
    };


protected:
    bool filled = false;
    //int calculate_column( array_d index, int dim){return index[dim];};
};
}
