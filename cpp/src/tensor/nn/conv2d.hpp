#pragma once

#define USE_CONV_NAIVE 0

#if USE_CONV_NAIVE
#include "conv2d_naive.hpp"
namespace ts {
    using namespace naive;
}
#else
#include "conv2d_im2col.hpp"
namespace ts {
    using namespace im2col;
}
#endif