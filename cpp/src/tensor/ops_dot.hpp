#pragma once
#include "tensor_forward.hpp"
#include <vector>

namespace ts {

auto dot(MatrixF const &, VectorF const &) -> VectorF;

auto dot(MatrixF const & A, MatrixF const & B, bool A_T=false, bool B_T=false) -> MatrixF ;

auto dot(Tensor<float, 3> const & A, MatrixF const & B) -> Tensor<float, 3>;

}