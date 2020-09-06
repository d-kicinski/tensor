#include "matrix.hpp"

namespace space {

Matrix::Matrix(unsigned int m, unsigned int n)
{
    _shape = {m, n};
    _data.resize(m * n);
}

auto multiply(const Matrix &lhs, const Matrix &rhs) -> Matrix
{
    Matrix result(lhs);
    return result;
}

auto operator+(Matrix const &lhs, Matrix const &rhs) -> Matrix
{
    if (lhs.shape() != rhs.shape()) {
        throw InvalidShapeException();
    }
    Matrix result(lhs);
    std::transform(std::begin(lhs), std::end(lhs), std::begin(lhs), std::begin(result),
                   std::plus<>());
    return result;
}

auto Matrix::operator[](const std::vector<int> &position) -> float &
{
    auto index = position[0] * _shape[1] + position[1];
    return _data[index];
}

auto Matrix::operator[](int index) -> Matrix::Vector
{
    if (index >= _shape[0]) {
        throw IndexOutOfRangeException(index);
    }
    auto begin = std::begin(_data) + (index * _shape[1]);
    auto end = std::begin(_data) + ((index + 1) * _shape[1]);

    return Vector(begin, end);
}

bool Matrix::operator==(const Matrix &rhs) const
{
    return std::tie(_data, _shape) == std::tie(rhs._data, rhs._shape);
}

Matrix::Matrix(Matrix::ShapeVector shape)
{
    _shape = std::move(shape);
    _data.resize(std::accumulate(std::begin(shape), std::end(shape), 0, std::multiplies<>()));
}

Matrix::Matrix(std::initializer_list<Vector> initializer)
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
    _shape = {static_cast<unsigned int>(m), static_cast<unsigned int>(n)};
}

auto Matrix::zeros(int m, int n) -> Matrix { return std::move(Matrix(m, n)); }

auto Matrix::ones(int m, int n) -> Matrix
{
    Matrix matrix(m, n);
    std::fill(std::begin(matrix), std::end(matrix), 1);
    return std::move(matrix);
}

//auto Matrix::operator=(std::initializer_list<Vector> initializer) -> Matrix
//{
//    //    return std::move(*thisinitializer));
//}
} // namespace space
