#pragma once
#include "tensor_forward.hpp"

namespace ts {

auto multiply(Matrix, Vector) -> Vector;

auto multiply(Matrix, Matrix) -> Matrix ;

}