## How-to-Use

### Downloads
The repository can be cloned as follows:
```bash
git clone https://github.com/DevelopNumericalLibraryForSupercomputer/SparseEigensolver.git
```
Alternatively, the source files can be downloaded through GitHub 'Download ZIP' option.


### Compile

The stable version is in the **stable** branch.

#### Prerequisites
The following prerequisites are required to compile SparseEigensolver:
* C++ 11 or higher
* OpenMP (`-lgomp` or `-qopenmp`)
* Intel(R) Math kernel library - BLAS, LAPACK, PBLAS (`-mkl=parallel`)
* Python3 with NumPy (`-lpython3.12`, `-I/PATH/OF/PYTHON/INCLUDE/python3.12`, `-I/PATH/OF/NUMPY/INCLUDE`, `-L/PATH/OF/PYTHON/LIBRARIES/lib/`)
* [YAML-cpp 0.6.0](https://github.com/jbeder/yaml-cpp) (`-lyaml-cpp`)

#### Example compile lines
##### Nurion (Korea Supercomputing Center)
* Specification of Nurion cluster: https://www.ksc.re.kr/eng/resources/nurion

```bash
module load craype-mic-knl intel/oneapi_21.3 impi/oneapi_21.3
mpiicpc -qopenmp -g -std=c++17 -mkl=parallel -o exe cpu_serial_test.cpp -L/apps/compiler/intel/oneapi_21.3/mkl/2021.3.0/lib/intel64/ -I/PATH/OF/PYTHON/INCLUDE/python3.12/ -I/PATH/OF/NUMPY/INCLUDE/ -I/PATH/OF/SparseEigensolver/include/ -L/PATH/OF/PYTHON/LIBRARIES/lib/ -lpython3.12
```

### Usage

SparseEigensolver is C++ based header-only library. We also provide python interface of eigenvalue decomposition function.

#### Folders
- **include**: Contains all project header files and python wrappers
- **example** : Example cpp files for test run

#### Structure of the library 

To use this library, the following aspects need to be determined:

- **Comm class** - The method of acceleration: *MKL* (Serial), *MPI*, *CUDA*, ...
- **Map class** - The distribution and storage method for matrix data: *Contiguous1D*, *BlockCycling*
- **Tensor class** - The type of matrix data: *DenseTensor* or *SparseTensor*

#### Example 1. serial_matrix_operations.cpp

Let's start with a basic example of tensor creation.
All the classes and functions used in this example are defined within the `SE` namespace.
The `Tensor` class requires pre-defined objects of the `Comm` and `Map` classes.
For serial CPU execution, you need to use `MKLComm`.
Since we will not distribute the data for the tensor because we are performing calculations in a serial manner, you can use the simplest map,  `Contiguous1DMap`.

Therefore, you should include the following headers:
```cpp
#include "device/mkl/TensorOp.hpp"
#include "Contiguous1DMap.hpp"
#include "DenseTensor.hpp"
#include "SparseTensor.hpp"
```
Additionally, you need to include `DenseTensor.hpp` and `SparseTensor.hpp` to consturct a tensor.

Now, let's create a tensor. First, create a `Comm` object using the `CommInp` class.

```cpp
using namespace SE;

MKLCommInp comm_inp;
std::unique_ptr< SE::Comm < SE::DEVICETYPE::MKL > > ptr_comm = comm_inp.create_comm();
```

To create `Tensor`, we first make a `std::array` that indicates the shape of the tensor. After that, use `MapInp` to create `ptr_map`.
```cpp
std::array<int, 2> test_shape = {4,3};
Contiguous1DMapInp<2> map_inp( test_shape );
std::unique_ptr<Map<2,MTYPE::Contiguous1D> > ptr_map = map_inp.create_map();
```

Fianlly, The `DenseTensor` can be created by using constructor of `DenseTensor`:
```cpp
SE::DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix(ptr_comm,ptr_map);
```

If you want to insert the value, use `global_insert_value` or `local_insert_value` function.
If you already prepared the data of the matrix, you can put data in the constructor too.

```cpp
int N = 20;
std::array<int, 2> test_shape2 = {N,N};
Contiguous1DMapInp<2> map2_inp( test_shape2 );
std::unique_ptr<double[], std::function<void(double*)> > test_data2 ( malloc<double, DEVICETYPE::MKL>(N*N), free<DEVICETYPE::MKL> );
// 3th order kinetic energy matrix
for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
        test_data2.get()[i+j*N] = 0;
        if(i == j)                  test_data2.get()[i+j*N] -= 5.0/2.0;
        if(i == j +1 || i == j -1)  test_data2.get()[i+j*N] += 4.0/3.0;
        if(i == j +2 || i == j -2)  test_data2[i+j*N] -= 1.0/12.0;
    }
}
DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix2(ptr_comm, map2_inp.create_map(), std::move(test_data2));
``` 

`SparseTensor` can be created in a similar manner. In the following example, the last parameter should be slightly larger than the approximate number of data points. The class will allocate memory for `N*3` data points, for example. Please note that you must call the `complete()` function after the sparse tensor construction is finished.

```cpp
SE::SparseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> test_sparse( ptr_comm, map2_inp.create_map(), N*3);
for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
        std::array<int,2> index = {i,j};
        if(i == j)                   test_sparse.global_insert_value(index, - 5.0/2.0);
        if(i == j +1 || i == j -1)   test_sparse.global_insert_value(index, 4.0/3.0);
        if(i == j +2 || i == j -2)   test_sparse.global_insert_value(index, (-1.0)/12.0);
    }
}
test_sparse.complete();
```

You can always print out the map and tensor information using `std::cout`:

```cpp
std::cout << *ptr_map <<std::endl;
std::cout << test_sparse << std::endl; 
```

To perform operations such as matrix-matrix multiplication or QR decomposition, you can use functions from `TensorOp`.

```cpp
auto copy_matrix(test_matrix2);
TensorOp::orthonormalize(copy_matrix,"qr");
std::cout << copy_matrix << std::endl;
auto result_matrix = TensorOp::matmul( test_sparse, test_matrix2, TRANSTYPE::N, TRANSTYPE::N);
```

For additional explanations about the example file (`serial_matrix_operations.cpp`), please refer to the comments within the file and the corresponding output (`serial_matrix_operations.output`).

#### Example2. cpu_serial_diagonalization.cpp
To perform matrix diagonalization, you need to decide whether you want to use direct diagonalization (for `DenseTensor` only) or iterative diagonalization.
If the user provides a method to perform matrix-vector multiplication (`TensorOperation`) in any form, we can use it to converge the lowest few eigenvalues of the given matrix to the desired accuracy using Davidson diagonalization.
In Davidson diagonalization, you can perform diagonalization without explicitly providing the entire matrix ($A$), as long as you can compute the product $Ax$ for guess vectors $x$.

In this library, users can provide a method for performing matrix-vector multiplication in the following ways:

- Provide the entire matrix (`DenseTensor`, `SparseTensor`)
- Implement the matrix-vector multiplication directly using the functions from `SE::TensorOp`
- Implement the matrix-vector multiplication using **Python** and `Numpy`.

#### Diagonalization options
Various options required for iterative diagonalization can be loaded from an external YAML file.
The following is an example of an options file.

```yaml
solver_options:
  algorithm: Davidson  # Specify the algorithm (Davidson / Direct)
  max_iterations: 1000  # Maximum number of iterations
  tolerance: 1e-6      # Convergence tolerance
  max_block: 2         # Maximum number of block expansion for each iteration

matrix_options:
  matrix_type: RealSym  # Real, RealSym, Complex, Hermitian

eigenvalue_options:
  num_eigenvalues: 3  # Number of eigenvalues to compute

preconditioner_options:
  preconditioner_type: Diagonal  # Specify the preconditioner type (e.g., Diagonal, ISI2)
  preconditioner_tolerance: 1e-3  # Tolerance for preconditioner
  preconditioner_max_iterations: 30  # max iteration number for preconditioner

locking_options:
  use_locking: false # Enable or disable locking
```
Further explanations can be found in [ExaTQ webpage](https://www.exatq.net/)
