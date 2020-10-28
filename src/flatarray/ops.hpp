#pragma once
#include "flatarray.hpp"

#ifdef USE_BLAS
    #include "ops_blas.hpp"
#else
    #include "ops_naive.hpp"
#endif

using Matrix = FlatArray<float, 2>;
using Vector = FlatArray<float, 1>;
