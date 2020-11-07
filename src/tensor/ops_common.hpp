#pragma once
#include "tensor.hpp"
#include <algorithm>

namespace ts {

template <typename Element, int Dim>
auto add(Tensor<Element, Dim> t1, Tensor<Element, Dim> t2) -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(t1);
    std::transform(std::begin(t2), std::end(t2), std::begin(t2), std::begin(result), std::plus<>());
    return result;
}

}