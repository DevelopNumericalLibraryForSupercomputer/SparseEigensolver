// PyTensorOperations.hpp
#pragma once
#include "decomposition/TensorOperations.hpp"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include <string>

namespace SE{

//typedef void   (*MatrixOneVecCallback)(const double* input_vec, double* output_vec, const int size, void* user_data);
//typedef void   (*MatrixMultVecCallback)(const double* input_vecs, double* output_vec, const int num_vec, const int size, void* user_data);
//typedef double (*GetDiagElementCallback)(int index, void* user_data);
//typedef void   (*GetGlobalShapeCallback)(int* shape, void* user_data);

template<MTYPE mtype, DEVICETYPE device>
class PyTensorOperations: public TensorOperations<mtype, device>{
private:
    void MatrixOneVec_wrapper(double* input_vec, double* return_vec, int size) const;
    void MatrixMultVec_wrapper(double* input_vec, double* return_vec, int num_vec, int size) const;
    double GetDiagElement_wrapper(int index) const;
    void GetGlobalShape_wrapper(int* shape) const;

    void initialize_numpy() const{
        if (_import_array() < 0) {
            PyErr_Print();
            throw std::runtime_error("numpy.core.multiarray failed to import");
        }
    };
    std::string filename;
    std::string filepath;
public:
    PyTensorOperations(){
        this->filename = "tensor_operations";
        this->filepath = "";
    };
    PyTensorOperations(std::string filename){
        set_python_pathway(filename);
    }
    DenseTensor<1, double, mtype, device> matvec(const DenseTensor<1, double, mtype, device>& vec) const override{
        auto return_vec = DenseTensor<1, double, mtype, device>(vec);
        int size = vec.ptr_map->get_global_shape(0);
        //double* input_vec = vec.copy_data();
        MatrixOneVec_wrapper(vec.data.get(), return_vec.data.get(), size);

        return return_vec;
    }
    DenseTensor<2, double, mtype, device> matvec(const DenseTensor<2, double, mtype, device>& vec) const override{
        auto return_vec = DenseTensor<2, double, mtype, device>(vec);
        int size = vec.ptr_map->get_global_shape(0);
        int num_vec = vec.ptr_map->get_global_shape(1);
        //double* input_vec = vec.copy_data();
        MatrixMultVec_wrapper(vec.data.get(), return_vec.data.get(), num_vec, size);

        return return_vec;
    }
    double get_diag_element(const int index) const override{
        return GetDiagElement_wrapper(index);
    }
    std::array<int, 2> get_global_shape() const override{
        int shape[2];
        GetGlobalShape_wrapper(shape);
        return {static_cast<int>(shape[0]), static_cast<int>(shape[1])};
    }
    void set_python_pathway(std::string new_filename){
        int pos = new_filename.rfind('/');
        
        if(pos != std::string::npos){
            this->filepath = new_filename.substr(0, pos);
            this->filename = new_filename.substr(pos+1);
        }
        else{
            this->filename = new_filename;
            this->filepath = "";
        }
        if( this->filename.size() >= 3 && this->filename.compare(this->filename.size()-3, 3, ".py") == 0){
            this->filename = this->filename.substr(0, this->filename.size()-3);
        }
        else{
            //this->filename = this->filename;
        }
    }


};

template<MTYPE mtype, DEVICETYPE device>
void PyTensorOperations<mtype, device>::MatrixOneVec_wrapper(double* input_vec, double* return_vec, int size) const{
    MatrixMultVec_wrapper(input_vec, return_vec, 1, size);
};

template<MTYPE mtype, DEVICETYPE device>
void PyTensorOperations<mtype, device>::MatrixMultVec_wrapper(double* input_vec, double* return_vec, int num_vec, int size) const{
    Py_Initialize();
    initialize_numpy(); // Initialize NumPy API

    // Add the directory containing EOMCC.py to the Python path
    PyRun_SimpleString("import sys");
    if(filepath.size() > 0){
        PyRun_SimpleStringFlags(("sys.path.append('" + filepath + "')").c_str(), nullptr);
    }

    // Import the Python module containing the function
    PyObject* pName = PyUnicode_DecodeFSDefault(filename.c_str());
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the function from the module
        PyObject* pFunc = PyObject_GetAttrString(pModule, "matvec");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Create a NumPy array from the C++ array
            npy_intp dims[2] = {size, num_vec};
            PyObject* pArray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, input_vec);

            if (!pArray) {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_Finalize();
                throw std::runtime_error("Failed to create numpy array");
            }
            // Call the Python function
            PyObject* pValue = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
            Py_DECREF(pArray);

            if (pValue != nullptr) {
                // Check if the returned value is a NumPy array
                if (PyArray_Check(pValue)) {
                    PyArrayObject* npArr = (PyArrayObject*)pValue;
                    double* data = (double*)PyArray_DATA(npArr);
                    std::copy(data, data + size * num_vec, return_vec);
                } else {
                    Py_DECREF(pValue);
                    Py_DECREF(pFunc);
                    Py_DECREF(pModule);
                    Py_Finalize();
                    throw std::runtime_error("Python function did not return a numpy array");
                }

                Py_DECREF(pValue);
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_Finalize();
            } else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                Py_Finalize();
                throw std::runtime_error("Failed to call Python function");
            }
        } else {
            PyErr_Print();
            Py_DECREF(pModule);
            Py_Finalize();
            throw std::runtime_error("Python function not callable");
        }
    } else {
        PyErr_Print();
        Py_Finalize();
        throw std::runtime_error("Failed to load Python module");
    }
};

template<MTYPE mtype, DEVICETYPE device>
double PyTensorOperations<mtype, device>::GetDiagElement_wrapper(int index) const{
    Py_Initialize();
    // Add the directory containing EOMCC.py to the Python path
    PyRun_SimpleString("import sys");
    if(filepath.size() > 0){
        PyRun_SimpleStringFlags(("sys.path.append('" + filepath + "')").c_str(), nullptr);
    }

    // Import the Python module containing the function
    PyObject* pName = PyUnicode_DecodeFSDefault(filename.c_str());
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the function from the module
        PyObject* pFunc = PyObject_GetAttrString(pModule, "get_diag_element");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Convert the C++ int to Python int
            PyObject* pValueIndex = PyLong_FromLong(index);
            // Call the Python function
            PyObject* pValue = PyObject_CallFunctionObjArgs(pFunc, pValueIndex, NULL);
            Py_DECREF(pValueIndex);
            if (pValue != nullptr) {
                // Convert the Python result back to C++ double
                double result = PyFloat_AsDouble(pValue);
                Py_DECREF(pValue);
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_Finalize();
                return result;
            } else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                Py_Finalize();
                throw std::runtime_error("Failed to call Python function");
            }
        } else {
            PyErr_Print();
            Py_DECREF(pModule);
            Py_Finalize();
            throw std::runtime_error("Python function not callable");
        }
    } else {
        PyErr_Print();
        Py_Finalize();
        throw std::runtime_error("Failed to load Python module");
    }

};

template<MTYPE mtype, DEVICETYPE device>
void PyTensorOperations<mtype, device>::GetGlobalShape_wrapper(int* shape) const{
    Py_Initialize();

    // Add the directory containing EOMCC.py to the Python path
    PyRun_SimpleString("import sys");
    if(filepath.size() > 0){
        PyRun_SimpleStringFlags(("sys.path.append('" + filepath + "')").c_str(), nullptr);
    }

    // Import the Python module containing the function
    PyObject* pName = PyUnicode_DecodeFSDefault(filename.c_str());
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the function from the module
        PyObject* pFunc = PyObject_GetAttrString(pModule, "get_global_shape");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Call the Python function (no arguments)
            PyObject* pValue = PyObject_CallObject(pFunc, NULL);

            if (pValue != nullptr) {
                // Check if the returned value is a tuple
                if (PyTuple_Check(pValue) && PyTuple_Size(pValue) == 2) {
                    shape[0] = PyLong_AsLong(PyTuple_GetItem(pValue, 0));  // Rows
                    shape[1] = PyLong_AsLong(PyTuple_GetItem(pValue, 1));  // Columns
                } else {
                    Py_DECREF(pValue);
                    Py_DECREF(pFunc);
                    Py_DECREF(pModule);
                    Py_Finalize();
                    throw std::runtime_error("Python function did not return a tuple of size 2");
                }

                Py_DECREF(pValue);
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_Finalize();
            } else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                Py_Finalize();
                throw std::runtime_error("Failed to call Python function");
            }
        } else {
            PyErr_Print();
            Py_DECREF(pModule);
            Py_Finalize();
            throw std::runtime_error("Python function not callable");
        }
    } else {
        PyErr_Print();
        Py_Finalize();
        throw std::runtime_error("Failed to load Python module");
    }
}


} 
