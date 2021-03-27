#pragma once
#include "tensor_forward.hpp"
#include <vector>

#ifdef USE_BLAS
#include "ops_dot_blas.hpp"
namespace ts {
using namespace blas;
}
#else
#include "ops_dot_naive.hpp"
namespace ts {
using namespace naive;
}
#endif

#if BUILD_BENCHMARK
#include "ops_dot_naive.hpp"
#endif
