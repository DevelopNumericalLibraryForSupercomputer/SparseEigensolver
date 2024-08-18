#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>

// Function to get the global shape of the matrix from Python
void getglobalshape(int* shape) {
    Py_Initialize();
    //import_array();  // Initialize NumPy API

    // Add the directory containing EOMCC.py to the Python path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/jaewook/projects/SparseEigensolver/python/callback_test2')");  // Replace with the correct path to your module

    // Import the Python module containing the function
    PyObject* pName = PyUnicode_DecodeFSDefault("EOMCC");
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

int main() {
    int shape[2];  // Array to hold the shape [rows, cols]

    try {
        getglobalshape(shape);
        std::cout << "Matrix shape: " << shape[0] << " x " << shape[1] << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
