#include <Python.h>
#include <iostream>

// Function to call the Python matvec function
void matvec(double* arr, int rows, int cols, double* result) {
    Py_Initialize();

    // Add the directory containing EOMCC.py to the Python path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/path/to/your/module')");  // Replace with the correct path to your module

    // Import the Python module containing the function
    PyObject* pName = PyUnicode_DecodeFSDefault("EOMCC");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the function from the module
        PyObject* pFunc = PyObject_GetAttrString(pModule, "matvec");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Convert C++ double* array to Python list of lists
            PyObject* pArgs = PyList_New(rows);
            for (int i = 0; i < rows; ++i) {
                PyObject* pSubList = PyList_New(cols);
                for (int j = 0; j < cols; ++j) {
                    PyList_SetItem(pSubList, j, PyFloat_FromDouble(arr[i * cols + j]));
                }
                PyList_SetItem(pArgs, i, pSubList);
            }

            // Call the Python function
            PyObject* pValue = PyObject_CallObject(pFunc, PyTuple_Pack(1, pArgs));
            Py_DECREF(pArgs);

            if (pValue != nullptr) {
                // Convert Python list back to C++ double* array
                for (int i = 0; i < rows; ++i) {
                    PyObject* pSubList = PyList_GetItem(pValue, i);
                    for (int j = 0; j < cols; ++j) {
                        result[i * cols + j] = PyFloat_AsDouble(PyList_GetItem(pSubList, j));
                    }
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
    int rows = 2;
    int cols = 3;
    double arr[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double result[6];

    try {
        matvec(arr, rows, cols, result);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << result[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
