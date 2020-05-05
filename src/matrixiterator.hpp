#pragma once

#include "matrix.hpp"
#include "matrixview.hpp"

namespace space {
class Matrix;

class MatrixIterator {
    int _num = 0;
    Matrix const &_matrix;
  public:
    MatrixIterator();
    MatrixIterator(Matrix const &matrix, long num = 0);
    auto operator++() -> MatrixIterator &;
    auto operator++(int) -> MatrixIterator;
    bool operator==(MatrixIterator other) const;
    bool operator!=(MatrixIterator other) const;
    MatrixView operator*();

    // iterator traits
    using difference_type = int;
    using value_type = MatrixView;
    using pointer = MatrixView const *;
    using reference = MatrixView const &;
    using iterator_category = std::forward_iterator_tag;
};
} // namespace space
