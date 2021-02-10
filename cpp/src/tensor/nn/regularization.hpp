#pragma once

#include <tensor/tensor.hpp>

namespace ts {

auto l2(std::vector<MatrixF> weights, float alpha=1e-3) -> float
{
    return std::transform_reduce(weights.begin(), weights.end(), 0.0,
                                 std::plus<>(),
                                 [&](auto & tensor) {
                                   return 0.5 * alpha * ts::sum(ts::pow(tensor, 2));
                                 });
}

}
