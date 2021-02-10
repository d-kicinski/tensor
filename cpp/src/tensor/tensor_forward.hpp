#pragma once

namespace ts {

template <typename Element, int Dim> class Tensor;

// Convenient typedefs
template<typename Element>
using Matrix = Tensor<Element, 2>;
using MatrixF = Matrix<float>;
using MatrixI = Matrix<int>;

template<typename Element>
using Vector = Tensor<Element, 1>;
using VectorF = Vector<float>;
using VectorI = Vector<int>;


}
