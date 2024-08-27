
### SparseEigensolver

SparseEigensolver is a library for iterative diagonalization of dense/sparse matrices using the Davidson algorithm. This library is designed to efficiently handle cases where the matrix is dense, sparse, or too complex to be fully defined, requiring on-the-fly matrix-vector multiplication. It supports MPI parallelization for all these scenarios, and plans to add CUDA and ROCm support in future updates.

#### Davidson Algorithm

The Davidson algorithm is a powerful iterative method commonly used to find the lowest eigenvalues and corresponding eigenvectors of large, sparse matrices.  Its strength lies in its ability to efficiently converge to the desired eigenvalues by iteratively and selectively expanding the subspace, which reduces the overall computational cost compared to direct methods.

#### Advantages of the Davidson Algorithm

- **Efficiency on Sparse Matrices**: The Davidson algorithm excels on large sparse matrices where direct methods would be computationally prohibitive.
- **Iterative Convergence**: It provides a way to iteratively converge to the smallest eigenvalues, which is particularly useful in quantum chemistry and physics applications.
- **Scalability**: The algorithm can be easily parallelized, making it suitable for high-performance computing environments.

### Features

1. **Block-Davidson iterative diagonalization**: The library uses the Block-Davidson method, which is based on given guess vectors, to iteratively find the multiple eigenvalues and eigenvectors.

2. **Flexible matrix representation**: Users can represent matrices as `DenseTensor`, `SparseTensor` (COO format), or even implement custom matrix-vector multiplication functions (see `TestOperations.hpp` for examples).

3. **ISI2 Preconditioner**: The library significantly reduces the number of iterations required by using the ISI2 preconditioner (accepted, doi: 10.1021/acs.jctc.4c00721).

4. **Customizable diagonalization options**: The library allows flexible customization of diagonalization options using the YAML format (feature under construction).

5. **Python interface**: The library provides two types of Python interfaces:
   1) **Import matrix-vector multiplication code from Python**: To use custom matrix-vector multiplication, users need to implement the code in C++. To simplify this, the library uses the Python/C API to allow the library to read and use matrix-vector multiplication code written in Python using NumPy. (See `include/decomposition/tensor_operations.py` for a reference).
   2) **Decompose function Python interface**: The core `decompose` function, which performs the diagonalization, has a Cython wrapper, allowing it to be called directly from Python. (This feature is under construction, and installation instructions will be added soon).


6. **Future CUDA and ROCm support**: The library will support CUDA and ROCm to further improve its performance on modern GPU architectures.


### Authors

- **Sunghwan Choi** (sunghwanchoi@inha.ac.kr) - Department of Chemistry, Inha University
- **Jaewook Kim** (jaewookkim@kisti.re.kr) - Department of High-Fidelity Model Acceleration Research, Korea Institute of Science and Technology Information (KISTI)

#### Citation

If you refer to this project, please use the following BibTeX entry:

```bibtex
@article{WOO2024,
title = {- ,
journal = {Journal of Chemical Theory and Computation},
volume = {-},
pages = {-},
year = {2024},
issn = {-},
doi = {https://doi.org/10.1021/acs.jctc.4c00721},
url = {https://doi.org/10.1021/acs.jctc.4c00721},
author = {Jeheon Woo and Woo Youn Kim and Sunghwan Choi}
}

