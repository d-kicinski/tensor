#pragma once
#include "tensor.hpp"

#ifdef USE_BLAS
    #include "ops_blas.hpp"
#else
    #include "ops_naive.hpp"
#endif

namespace ts {

using Matrix = Tensor<float, 2>;
using Vector = Tensor<float, 1>;

}
