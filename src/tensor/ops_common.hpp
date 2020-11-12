#pragma once
#include "tensor_forward.hpp"

namespace ts {

template <typename Element, int Dim>
auto add(Tensor<Element, Dim>, Tensor<Element, Dim>) -> Tensor<Element, Dim>;

template <typename Element, int Dim>
auto maximum(Element, Tensor<Element, Dim>) -> Tensor<Element, Dim>;

template <typename Element, int Dim>
auto mask(Tensor<Element, Dim>, std::function<bool(Element)>) -> Tensor<bool, Dim>;

template <typename Element, int Dim>
auto assign_if(Tensor<Element, Dim>, Tensor<bool, Dim>, Element) -> Tensor<Element, Dim>;

template <typename Element, int Dim>
auto multiply(Tensor<Element, Dim>, Element) -> Tensor<Element, Dim>;

auto transpose(Matrix const &) -> Matrix ;

}
