#include "dropout.hpp"

ts::Dropout::Dropout(float keep_probability) : _p(keep_probability) {}

auto ts::Dropout::operator()(const ts::MatrixF &input) -> ts::MatrixF { return forward(input); }

auto ts::Dropout::forward(const ts::MatrixF &input) -> ts::MatrixF
{
    // TODO: change input parameters in initialization.hpp to std::array
    int dim0 = input.shape(0);
    int dim1 = input.shape(1);
    _weight = ts::multiply(ts::bernoulli<float, 2>({dim0, dim1}, _p),
                           1.0f / (1.0f - _p));

    return ts::multiply(input, _weight);
}

auto ts::Dropout::backward(const ts::MatrixF &d_output) -> ts::MatrixF { return ts::multiply(_weight, d_output); }
