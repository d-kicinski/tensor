#pragma once

namespace ts {

template <typename Element, int Dim, bool AllocationFlag = true> class Tensor;

// Convenient typedefs
typedef Tensor<float, 2, true> Matrix;
typedef Tensor<float, 1, true> Vector;
typedef Tensor<float, 2, false> MatrixView;
typedef Tensor<float, 1, false> VectorView;

}
