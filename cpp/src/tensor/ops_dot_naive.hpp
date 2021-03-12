#pragma once
#include "tensor_forward.hpp"
#include <vector>

namespace ts::naive {

auto outer_product(VectorF const &, VectorF const &) -> MatrixF;

auto dot(VectorF const &, VectorF const &) -> float;

auto dot(MatrixF const &, VectorF const &) -> VectorF;

auto dot(MatrixF const &A, MatrixF const &B, bool A_T = false, bool B_T = false) -> MatrixF;

auto dot(Tensor<float, 3> const &A, MatrixF const &B) -> Tensor<float, 3>;

} // namespace ts::naive
