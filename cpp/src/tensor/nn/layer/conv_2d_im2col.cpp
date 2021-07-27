#include "conv_2d_im2col.hpp"
#include "tensor/nn/im2col.hpp"
#include "tensor/nn/initialization.hpp"
#include <tensor/nn/conv_2d.hpp>

ts::im2col::Conv2D::Conv2D(Variable<float, 2> weight, std::optional<Variable<float, 1>> bias, int kernel_size,
                           int stride, int pad, int dilatation, Activation activation)
    : _weight(std::move(weight)), _bias(std::move(bias)), _activation(Activations::get(activation)), _stride(stride),
      _kernel_size(kernel_size), _pad(pad), _dilatation(dilatation)
{
    register_parameters(_weight);
    if (_bias) {
        register_parameters(_bias.value());
    }
}

auto ts::im2col::Conv2D::create(int in_channels, int out_channels, int kernel_size, int stride, int pad, int dilatation,
                                Activation activation, bool use_bias) -> Conv2D
{
    std::vector<int> shape = {out_channels, kernel_size * kernel_size * in_channels};
    Variable<float, 2> weight(std::make_unique<MatrixF>(ts::kaiming_uniform<float, 2>(shape)),
                              std::make_unique<MatrixF>(ts::zeros<float, 2>(shape)), "Conv2D(weight)");
    std::optional<Variable<float, 1>> bias = std::nullopt;
    if (use_bias)
        bias = std::make_optional(Variable<float, 1>(std::make_unique<VectorF>(ts::bias_init<float, 1>({out_channels})),
                                                     std::make_unique<VectorF>(ts::zeros<float, 1>({out_channels})),
                                                     "Conv2D(bias)"));
    return Conv2D(std::move(weight), std::move(bias), kernel_size, stride, pad, dilatation, activation);
}

auto ts::im2col::Conv2D::operator()(ts::Tensor<float, 4> const &input) -> Tensor<float, 4> { return forward(input); }

auto ts::im2col::Conv2D::forward(ts::Tensor<float, 4> const &input) -> ts::Tensor<float, 4>
{
    _input = input;

    auto const im2col_buffer_shape = ts::im2col::im2col_buffer_shape({input.shape(1), input.shape(2), input.shape(3)},
                                                                     _kernel_size, _stride, _pad, _dilatation);
    if (_im2col_buffer.data() == nullptr || _im2col_buffer.shape() != im2col_buffer_shape) {
        _im2col_buffer = Tensor<float, 2>(im2col_buffer_shape);
    }
    auto output = ts::conv_2d_im2col(input, _weight.tensor(), _im2col_buffer, _kernel_size, _stride, _pad, _dilatation);
    if (_bias.has_value()) {
        for (int b = 0; b < output.shape(0); ++b) {
            ts::add_(output(b), _bias.value().tensor());
        }
    }
    if (_activation) {
        output = _activation.value()->forward(output);
    }
    return output;
}

auto ts::im2col::Conv2D::backward(ts::Tensor<float, 4> const &d_output) -> ts::Tensor<float, 4>
{
    auto d_output_(d_output); // cheap, not a deep copy
    if (_activation) {
        d_output_ = _activation.value()->backward(d_output_);
    }
    auto [d_input, d_weight] = ts::conv_2d_backward_im2col(_input, _weight.tensor(), _im2col_buffer, d_output_,
                                                           _kernel_size, _stride, _pad, _dilatation);
    _weight.grad() += d_weight;

    if (_bias.has_value()) {
        for (int b = 0; b < d_output.shape(0); ++b) {
            for (int i = 0; i < d_output.shape(1); ++i) {
                for (int j = 0; j < d_output.shape(2); ++j) {
                    ts::add_(_bias.value().grad(), d_output(b, i, j));
                }
            }
        }
    }
    return std::move(d_input);
}

auto ts::im2col::Conv2D::weight() -> ts::Variable<float, 2> & { return _weight; }

auto ts::im2col::Conv2D::bias() -> std::optional<std::reference_wrapper<ts::Variable<float, 1>>>
{
    if (_bias.has_value()) {
        return std::make_optional(std::ref(_bias.value()));
    } else {
        return std::nullopt;
    }
}

auto ts::im2col::Conv2D::weights() -> VectorRef
{
    std::vector<std::reference_wrapper<ts::GradHolder<float>>> vars;
    vars.emplace_back(std::ref(weight()));
    if (_bias.has_value()) {
        vars.emplace_back(std::ref(bias().value()));
    }
    return vars;
}
