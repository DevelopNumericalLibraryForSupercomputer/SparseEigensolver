#pragma once
#include <iostream>
#include <iomanip>
#include <string>
#include <complex>
#include <cassert>
#include "mkl_wrapper.hpp"
#include "Utility.hpp"

namespace TensorHetero{
template <typename datatype>
class Matrix{
private:
    size_t row = 0;
    size_t column = 0;
    datatype* pointer; //column major: pointer[i + row*j] = matrix(i,j)
    size_t pointer_size = 0;

    inline size_t get_index(size_t i, size_t j) const;

public:
    Matrix(){};
    Matrix(size_t row, size_t column);
    Matrix(size_t row, size_t column, datatype *tensor);
    ~Matrix(){};

    size_t get_row() const { return this->row;    }
    size_t get_col() const { return this->column; }
    void print_shape() const;
    void print_matrix() const;
    Matrix<datatype> conj() const;

    datatype&           operator[](size_t index);
    datatype&           operator()(const size_t& i, const size_t& j);

    Matrix<datatype>&   operator=(const Matrix<datatype>& matrix);
    Matrix<datatype>    operator*(const Matrix<datatype>& matrix) const;
    Matrix<datatype>    operator*(const datatype number) const;
    Matrix<datatype>    operator+(const Matrix<datatype>& matrix) const;
    Matrix<datatype>    operator-(const Matrix<datatype>& matrix) const;
    
    datatype* get_row_vector(size_t row);
    datatype* get_column_vector(size_t column);

    Matrix<datatype>& multiply(const Matrix<datatype>& matrix1, const Matrix<datatype>& matrix2, std::string trans1, std::string trans2);
};

template <typename datatype>
std::ostream& operator<<(std::ostream& os, Matrix<datatype>& M);

template <typename datatype>
inline size_t Matrix<datatype>::get_index(size_t i, size_t j) const
{
    return i + j * this->row;
}

template <typename datatype>
Matrix<datatype>::Matrix(size_t row, size_t column)
{
    this->row = row;
    this->column = column;
    pointer_size = row * column;
    pointer = new datatype[pointer_size];

}

template <typename datatype>
Matrix<datatype>::Matrix(size_t row, size_t column, datatype *tensor){
    this->row = row;
    this->column = column;
    this->pointer_size = row * column;
    pointer = tensor;
}

template <typename datatype>
void Matrix<datatype>::print_shape() const{
    std::cout << "(" << this->row << "-by-" << this->column << ")" << std::endl;
}

template <typename datatype>
void Matrix<datatype>::print_matrix() const{
    for(size_t i = 0; i < this->row; ++i){
        for(size_t j = 0; j < this->column; ++j){
            std::cout << this(i,j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <>
Matrix<double> Matrix<double>::conj() const{
    /*
    Matrix<double> return_matrix(this->row, this->column);
    return_matrix.pointer = this->pointer;
    */
    Matrix<double> return_matrix = *this;
    return return_matrix;
}

template <>
Matrix<std::complex<double>> Matrix<std::complex<double>>::conj() const{
    Matrix<std::complex<double>> return_matrix(this->row, this->column);
    #pragma omp parallel for
    for (size_t i=0; i<pointer_size; i++){
        return_matrix[i] = std::conj(this->pointer[i]);
    }
    return return_matrix;
}

template <typename datatype>
datatype& Matrix<datatype>::operator[](size_t index){
    return this->pointer[index];
}

template <typename datatype>
datatype& Matrix<datatype>::operator()(const size_t& i, const size_t& j){
    return this->pointer[this->get_index(i,j)];
}

template <typename datatype>
Matrix<datatype>& Matrix<datatype>::operator=(const Matrix<datatype>& matrix){
    this->pointer = matrix.pointer;
    this->row = matrix.row;
    this->column = matrix.column;
    this->pointer_size = matrix.pointer_size;

    return *this;
}

template <typename datatype>
Matrix<datatype> Matrix<datatype>::operator*(const Matrix<datatype>& matrix) const{
    //Matrix multiplication
    Matrix<datatype> return_matrix;
    return_matrix.multiply(*this, matrix, "NoT", "NoT");

    return return_matrix;
}

template <typename datatype>
Matrix<datatype> Matrix<datatype>::operator*(const datatype number) const{
    Matrix<datatype> return_matrix(this->row, this->column);
    #pragma omp parallel for
    for(size_t i=0; i<pointer_size; i++){
        return_matrix[i] = this->pointer[i] * number;
    }
    return return_matrix;
}

template <typename datatype>
Matrix<datatype> Matrix<datatype>::operator+(const Matrix<datatype> &matrix) const{
    assert(this->row == matrix.row && this->column == matrix.column);
    Matrix<datatype> return_matrix(this->row, this->column);
    #pragma omp parallel for
    for(size_t i = 0; i < pointer_size; ++i){
        return_matrix.pointer[i] = (this->pointer[i] + matrix.pointer[i]);
    }
    return return_matrix;
}

template <typename datatype>
Matrix<datatype> Matrix<datatype>::operator-(const Matrix<datatype> &matrix) const{
    assert(this->row == matrix.row && this->column == matrix.column);
    Matrix<datatype> return_matrix(this->row, this->column);
    #pragma omp parallel for
    for(size_t i = 0; i < pointer_size; ++i){
        return_matrix.pointer[i] = (this->pointer[i] - matrix.pointer[i]);
    }
    return return_matrix;
}

template <typename datatype>
datatype* Matrix<datatype>::get_row_vector(size_t row){
    datatype* return_vector = new datatype[this->column];
    for(size_t i=0; i<this->column; ++i){
        return_vector[i] = this->operator()(row, i);
    }
    return return_vector;
}

template <typename datatype>
datatype* Matrix<datatype>::get_column_vector(size_t column){
    datatype* return_vector = new datatype[this->row];
    for(size_t i=0; i<this->row; ++i){
        return_vector[i] = this->operator()(i, column);
    }
    return return_vector;
}

template <typename datatype>
Matrix<datatype>& Matrix<datatype>::multiply(
    const Matrix<datatype> &matrix1, const Matrix<datatype> &matrix2, std::string trans1, std::string trans2)
{
    /*(example)
    Matrix<double> Mat3;
    Mat3.multiply(Mat1, Mat2, "NoT", "T");
    T : transpose, NoT : not transpose
    If not specify, trans1 == trans2 == "NoT"
    */
    //input structure check
    if( ! (  (trans1 == "T" or trans1 == "NoT") && (trans2 == "T" or trans2 == "NoT") ) ){
        std::cout << "Matrix::multiply : Wrong input parameter!" << std::endl;
        std::cout << "Check the string variables! trans1 = " << trans1 << ", trans2 = " << trans2 << std::endl;
        exit(-1);
    }

    //matrix size
    size_t row1, column1, row2, column2;
    row1 = matrix1.get_row(); column1 = matrix1.get_col();
    row2 = matrix2.get_row(); column2 = matrix2.get_col();

    if(trans1 == "NoT"){  this->row = row1; }
        else{             this->row = column1; }
    if(trans2 == "NoT"){  this->column = column2; }
        else{             this->column = row2; }
    this->pointer_size = this->row * this-> column;
    this->pointer = new datatype[this->pointer_size];
    
    /*
    if(trans1 == "NoT"){  row1 = matrix1.get_row(); column1 = matrix1.get_col();  }
        else{             row1 = matrix1.get_col(); column1 = matrix1.get_row();  }
    if(trans2 == "NoT"){  row2 = matrix2.get_row(); column2 = matrix2.get_col();  }
        else{             row2 = matrix2.get_col(); column2 = matrix2.get_row();  }
      
    if(column1 != row2){
        std::cout << "Matrix::multiply : Check the matrix shape!" << std::endl;
        std::cout << "( " << row1 << " x " << column1 << " ) X ( " << row2 << " x " << column2 << " ) is impossible!" << std::endl;
        exit(-1);
    }*/

    if(trans1 == "NoT" && trans2 == "NoT"){
        gemm<datatype>(CblasColMajor, CblasNoTrans, CblasNoTrans, row1, column2, column1,
             1.0, matrix1.pointer, row1, matrix2.pointer, row2, 0.0, this->pointer, this->get_row());
    }
    else if(trans1 == "NoT" && trans2 == "T"){
        gemm<datatype>(CblasColMajor, CblasNoTrans, CblasTrans, row1, row2, column1,
             1.0, matrix1.pointer, row1, matrix2.pointer, row2, 0.0, this->pointer, this->get_row());
    }
    else if(trans1 == "T" && trans2 == "NoT"){
        gemm<datatype>(CblasColMajor, CblasTrans, CblasNoTrans, column1, column2, row1,
             1.0, matrix1.pointer, row1, matrix2.pointer, row2, 0.0, this->pointer, this->get_row());
    }
    else if(trans1 == "T" && trans2 == "T"){
        gemm<datatype>(CblasColMajor, CblasTrans, CblasTrans, column1, row2, row1,
             1.0, matrix1.pointer, row1, matrix2.pointer, row2, 0.0, this->pointer, this->get_row());
    }
    else{
        std::cout << "Matrix::multiply : under test" << std::endl;
        exit(-1);
    }
    return *this;
}

template <typename datatype>
std::ostream& operator<<(std::ostream &os, Matrix<datatype> &M)
{
    os << std::endl;
    for(size_t i = 0; i < M.get_row(); ++i){
        for(size_t j = 0; j < M.get_col(); ++j){
            os << std::setw(10) << M(i,j) << " ";
        }
        os << std::endl;
    }
    return os;
}

/*
template <typename datatype>
Matrix<datatype> Map(std::vector<datatype> &tensor, size_t row, size_t column)
{
    Matrix<datatype> mapped_matrix(row, column, tensor);
    return mapped_matrix;

    // TODO: 여기에 return 문을 삽입합니다.
}
*/
}
