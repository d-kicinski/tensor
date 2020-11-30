%module tensor
%{
#include <tensor/tensor.hpp>
%}

/* Parse the header file to generate wrappers */
%include <std_vector.i>
%include <std_array.i>

%template(VectorInt) std::vector<int>;
%template(ArrayI2) std::array<int, 2>;

%include "../include_swig/tensor/tensor.hpp"
%template(TensorF2) ts::Tensor<float, 2>;
