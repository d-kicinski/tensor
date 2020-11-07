#pragma once

namespace ts {

template <typename Element, int Dim, bool AllocationFlag = true> class Tensor;

// Convenient typedefs
typedef Tensor<float, 2> Matrix;
typedef Tensor<float, 1> Vector;

}
