#pragma once

#ifdef USE_BLAS
    #include "ops_blas.hpp"
#else
    #include "ops_naive.hpp"
#endif

#include "ops_common.hpp"
