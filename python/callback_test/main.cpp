#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <stdexcept>

typedef void (*c_matvec_wrapper_t)(double*, int, int, double*);

void check_python_error() {
    if (PyErr_Occurred()) {
        PyErr_Print();
        throw std::runtime_error("Python error occurred");
    }
}

int main() {
    try {
        Py_Initialize();
        std::cout << "Python initialized" << std::endl;

        import_array(); // NumPy 초기화

        PyObject* sysPath = PySys_GetObject("path");
        PyObject* programName = PyUnicode_DecodeFSDefault(".");
        PyList_Append(sysPath, programName);
        Py_DECREF(programName);
        std::cout << "Added current directory to Python path" << std::endl;

        PyObject* pName = PyUnicode_DecodeFSDefault("matvec");
        std::cout << "Attempting to import module: matvec" << std::endl;
        PyObject* pModule = PyImport_Import(pName);
        Py_DECREF(pName);

        if (pModule == NULL) {
            check_python_error();
        }
        std::cout << "Module imported successfully" << std::endl;

        PyObject* pFunc = PyObject_GetAttrString(pModule, "matvec");
        if (pFunc == NULL) {
            check_python_error();
        }
        std::cout << "Got py_c_matvec_wrapper function" << std::endl;

        std::vector<double> arr = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        int rows = 2;
        int cols = 3;

        npy_intp dims[2] = {rows, cols};
        PyObject* pArr = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, arr.data());
        if (pArr == NULL) {
            check_python_error();
        }

        PyObject* pResult = PyObject_CallFunctionObjArgs(pFunc, pArr, NULL);
        if (pResult == NULL) {
            check_python_error();
        }

        double* result_data = (double*)PyArray_DATA((PyArrayObject*)pResult);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << result_data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }

        Py_DECREF(pArr);
        Py_DECREF(pResult);
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        Py_Finalize();
        std::cout << "Python finalized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}