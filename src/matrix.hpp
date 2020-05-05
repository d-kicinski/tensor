#pragma once

#include "exceptions.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <numeric>
#include <vector>

#include "matrixiterator.hpp"

namespace space {

class Matrix {
  public:
    using Vector = std::vector<float>;
    using ShapeVector = std::vector<unsigned int>;

  public:
    //    typedef MatrixIterator iterator;

    Vector _data;
    ShapeVector _shape{};

    auto begin() -> Vector::iterator { return std::begin(_data); }
    auto begin() const -> Vector::const_iterator { return std::begin(_data); }

    auto end() -> Vector::iterator { return std::end(_data); }
    auto end() const -> Vector::const_iterator { return std::end(_data); }

    Matrix(unsigned int m, unsigned int n);

    Matrix(ShapeVector shape);

    Matrix(std::initializer_list<Vector> initializer);

    static auto zeros(int m, int n) -> Matrix;

    static auto ones(int m, int n) -> Matrix;

    auto operator=(std::initializer_list<Vector> initializer) -> Matrix;

    bool operator==(const Matrix &rhs) const;

    bool operator!=(const Matrix &rhs) const { return !(rhs == *this); }

    auto operator[](int index) -> Vector;

    auto operator[](std::vector<int> const &position) -> float &;

    friend auto operator+(Matrix const &lhs, Matrix const &rhs) -> Matrix;

    friend auto multiply(Matrix const &lhs, Matrix const &rhs) -> Matrix;

    auto shape() const -> ShapeVector const & { return _shape; }

    auto data() const -> Vector const & { return _data; }
};
}; // namespace space