#pragma once

#include <vector>
namespace space {

template <typename Type> class Matrix {
  public:
    typedef std::vector<Type> Vector;
    typedef std::vector<Vector> MatrixType;
    MatrixType _matrix;

    Matrix(int m, int n) { _matrix.resize(m, std::vector<Type>(n)); }
    explicit Matrix(MatrixType const & matrix) { _matrix(matrix); }
    Matrix(std::initializer_list<Vector> initializer) { _matrix = initializer; }
};
}; // namespace matrix
