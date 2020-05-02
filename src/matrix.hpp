#pragma once

#include "exceptions.hpp"
#include <array>
#include <vector>
namespace space {

template <typename Type = float> class Matrix {
  public:
    typedef std::vector<Type> Vector;
    typedef typename Vector::iterator iterator;
    typedef typename Vector::const_iterator const_iterator;
    typedef std::vector<int> ShapeVector;

    Vector _data;
    ShapeVector _shape{};

    auto begin() -> iterator { return std::begin(_data); }
    auto begin() const -> const_iterator { return std::cbegin(_data); }

    auto end() -> iterator { return std::end(_data); }
    auto end() const -> const_iterator { return std::cend(_data); }

    Matrix(int m, int n)
    {
        _shape = {m, n};
        _data.resize(m * n);
    };
    explicit Matrix(Vector const &data) { _data(data); }

    Matrix(std::initializer_list<Vector> initializer)
    {
        int m = initializer.size();
        int n = -1;

        for (Vector element : initializer) {
            if (n == -1) {
                n = element.size();
            } else if (n != element.size()) {
                throw InvalidShapeException();
            }
            _data.insert(std::end(_data), std::begin(element), std::end(element));
        }
        _shape = {m, n};
    }

    auto operator=(std::initializer_list<Vector> initializer) -> Matrix
    {
        return std::move(*this(initializer));
    }

    static auto zeros(int m, int n) -> Matrix { return std::move(Matrix(m, n)); }

    static auto ones(int m, int n) -> Matrix
    {
        Matrix matrix(m, n);
        std::fill(std::begin(matrix), std::end(matrix), 1);
        return std::move(matrix);
    }

    bool operator==(const Matrix &rhs) const
    {
        return std::tie(_data, _shape) == std::tie(rhs._data, rhs._shape);
    }

    bool operator!=(const Matrix &rhs) const { return !(rhs == *this); }

    auto operator[](int index) -> Vector
    {
        if (index >= _shape[0]) {
            throw IndexOutOfRangeException(index);
        }
        auto begin = std::begin(_data) + (index * _shape[1]);
        auto end = std::begin(_data) + ((index + 1) * _shape[1]);

        return Vector(begin, end);
    }

    auto operator[](std::vector<int> const &position) -> Type &
    {
        auto index = position[0] * _shape[1] + position[1];
        return _data[index];
    }

    [[nodiscard]] auto shape() const -> ShapeVector const & { return _shape; }
};
}; // namespace space