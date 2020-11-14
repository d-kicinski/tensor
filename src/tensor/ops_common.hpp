#pragma once
#include <functional>
#include "tensor_forward.hpp"

namespace ts {

template <typename Element, int Dim>
auto add(Tensor<Element, Dim> const &, Tensor<Element, Dim> const &) -> Tensor<Element, Dim>;

auto add(Matrix const &, Vector const &) -> Matrix ;

template <typename Element, int Dim>
auto maximum(Element, Tensor<Element, Dim> const &) -> Tensor<Element, Dim>;

template <typename Element, int Dim>
auto mask(Tensor<Element, Dim> const &, std::function<bool(Element)>) -> Tensor<bool, Dim>;

template <typename Element, int Dim>
auto assign_if(Tensor<Element, Dim> const &, Tensor<bool, Dim> const &, Element) -> Tensor<Element, Dim>;

template <typename Element, int Dim>
auto multiply(Tensor<Element, Dim> const &, Element) -> Tensor<Element, Dim>;

auto transpose(Matrix const &) -> Matrix;

auto sum(Matrix const &, int) -> Vector;

}
