#pragma once
#include <array>
#include <vector>
#include <memory>

#include "Comm.hpp"
#include "Map.hpp"
//#include "Contiguous1DMap.hpp"
//#include "decomposition/DecomposeResult.hpp"

namespace SE{


template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device, STORETYPE store> 
class Tensor{
using array_d = std::array<size_t, dimension>;
using INTERNALTYPE = typename std::conditional< store==STORETYPE::DENSE,  DATATYPE* , std::vector<std::pair<array_d, DATATYPE> > >::type; 

public:
    //const STORETYPE store = store;
    const Comm<device> comm;
    const MAPTYPE map;
    INTERNALTYPE data; 

    // constructor
    Tensor(){
        filled=false;
    }

    Tensor(const Comm<device>& comm, const MAPTYPE& map):comm(comm),map(map){
        filled=false;
    }
    //Tensor(const Comm<device>& comm, const MAPTYPE& map, size_t reserve_size): comm(comm), map(map) {//_internal_dataype data reserve};
    Tensor(const Comm<device>& comm, const MAPTYPE& map, INTERNALTYPE& data): comm(comm), map(map), data(data){
        filled=false;
    } 

    Tensor(const Tensor<dimension,DATATYPE,MAPTYPE,device,store>& tensor): comm(tensor.comm), map(tensor.map){ 
        data = tensor.copy_data();
        filled=false;
    }

    //destructor
    ~Tensor() {
        // Depending on the storage type, perform cleanup
        if constexpr (store == STORETYPE::DENSE) {
            // If the data is stored as a pointer, delete it
            if (data != nullptr) {
                delete[] data;
                data = nullptr;
            }
        } else {
            // If the data is stored as a vector, no cleanup needed
        }
    }

    // copy functions
    Comm<device>* copy_comm() const {
        return comm.clone();
    }

    MAPTYPE* copy_map() const {
        return map.clone();
    }

    virtual INTERNALTYPE copy_data() const=0;

    // clone function 
    virtual Tensor<dimension,DATATYPE,MAPTYPE,device,store>* clone(bool call_complete=false) const=0;

    // insert function (add value ) 
    virtual void global_insert_value(const array_d global_array_index, const DATATYPE value)=0;
    virtual void local_insert_value (const array_d local_array_index,  const DATATYPE value)=0;
    virtual void global_insert_value(const size_t global_index,        const DATATYPE value)=0;
    virtual void local_insert_value (const size_t local_index,         const DATATYPE value)=0;

    // set function 
    virtual void global_set_value(const array_d global_array_index, const DATATYPE value)=0;
    virtual void local_set_value (const array_d local_array_index,  const DATATYPE value)=0;
    virtual void global_set_value(const size_t global_index,        const DATATYPE value)=0;
    virtual void local_set_value (const size_t local_index,         const DATATYPE value)=0;


    // access function, it works only when filled==false
    DATATYPE operator()(const size_t local_index) const {
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
    friend std::ostream& operator<< (std::ostream& stream, const Tensor<dimension,DATATYPE,MAPTYPE,device,store>& tensor){
    //void print_tensor_info() const{
        if(tensor.comm.get_rank() == 0){
            stream << "========= Tensor Info =========" <<std::endl;
            stream << "dimension: " << dimension<< "\n" 
                   << "DATATYPE: "  << typeid(DATATYPE).name()
                   << "   shape: ("  ;
            for (auto shape_i : tensor.map.get_global_shape()){
                stream << shape_i << ",";
            }
            stream << ")\n"
                   << "MAPTYPE: "   << typeid(MAPTYPE).name() << "\n" 
                   << "Device: "    << (int) device <<"\n"
                   << "store: "     << (int) store <<  std::endl;   
        }
        return stream;
    };


protected:
    bool filled = false;
    //size_t calculate_column( array_d index, size_t dim){return index[dim];};
};
}
