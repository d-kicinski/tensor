%module tensor
%{
#include <tensor/tensor.hpp>
%}

/* Parse the header file to generate wrappers */
%include <std_vector.i>
%include "../src/tensor/tensor.hpp"
%template(TensorF2) ts::Tensor<float, 2>;
