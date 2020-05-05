#include "matrixiterator.hpp"

namespace space {

MatrixIterator::MatrixIterator(const Matrix &matrix, long num) : _matrix(matrix), _num(num) {}

auto MatrixIterator::operator++() -> MatrixIterator &
{
    _num += 1;
    return *this;
}

auto MatrixIterator::operator++(int) -> MatrixIterator
{
    MatrixIterator retval = *this;
    ++(*this);
    return retval;
}

bool MatrixIterator::operator==(MatrixIterator other) const { return _num == other._num; }

bool MatrixIterator::operator!=(MatrixIterator other) const { return !(*this == other); }

MatrixView MatrixIterator::operator*() { return MatrixView(); }
} // namespace space
