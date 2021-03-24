#pragma once

namespace ts {

using size_type = unsigned long int;

template <typename Element, int Dim> class Tensor;
template <typename Element> class DataHolder;

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
