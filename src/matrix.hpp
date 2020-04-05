#pragma once

#include <vector>
template <typename Type> class Matrix {
    std::vector<std::vector<Type>> _array;

  public:
    Matrix(int m, int n) { _array.resize(m, std::vector<Type>(n)); }
};
