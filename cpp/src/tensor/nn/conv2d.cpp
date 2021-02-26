#include "conv2d.hpp"
#include "functional.hpp"

ts::Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, Activation activation, bool use_bias)
    : _stride(stride), _kernel_size(kernel_size), _use_bias(use_bias)
{
    _activation = Activations::get(activation);
    _weight = ts::MatrixF::randn({kernel_size * kernel_size * in_channels, out_channels});
    if (_use_bias)
        _bias = ts::VectorF(in_channels);
}

auto ts::Conv2D::operator()(const ts::Tensor<float, 3> & input) -> Tensor<float, 3>
{
    return forward(input);
}

auto ts::Conv2D::forward(const ts::Tensor<float, 3> & input) -> ts::Tensor<float, 3>
{
    _input = input;
    auto output =  ts::conv_2d(input, _weight, _kernel_size, _stride);
    if (_use_bias) {
       output = ts::add(output, _bias) ;
    }
    if (_activation) {
        output = _activation.value()->forward(output);
    }
    return output;
}

auto ts::Conv2D::backward(const ts::Tensor<float, 3> & d_output) -> ts::Tensor<float, 3>
{
    auto d_output_(d_output); // cheap, not a deep copy
    if (_activation) {
        d_output_ = _activation.value()->backward(d_output_);
    }
    auto [d_input, d_weight] =  ts::conv_2d_backward(_input, _weight, d_output_, _kernel_size, _stride);
    _d_weight = std::move(d_weight);

    if (_use_bias) {
        for (int i = 0; i < d_output.shape(0); ++i) {
            for (int j = 0; j < d_output.shape(1); ++j) {
                ts::add_(_d_bias, d_output(i, j));
            }
        }
    }
    return std::move(d_input);
}

auto ts::Conv2D::update(float step_size) -> void
{
    _weight +=  ts::multiply(_d_weight, -step_size);
    _bias += ts::multiply(_d_bias, -step_size);
}

auto ts::Conv2D::weight() -> ts::MatrixF {
    return _weight;
}

auto ts::Conv2D::bias() -> ts::VectorF
{
    return _bias;
}
