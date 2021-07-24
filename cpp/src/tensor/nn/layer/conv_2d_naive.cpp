#include "conv_2d_naive.hpp"
#include "tensor/nn/initialization.hpp"
#include <tensor/nn/conv_2d.hpp>

ts::naive::Conv2D::Conv2D(Variable<float, 2> weight, std::optional<Variable<float, 1>> bias, int kernel_size,
                          int stride, Activation activation)
    : _weight(std::move(weight)), _bias(std::move(bias)), _activation(Activations::get(activation)), _stride(stride),
      _kernel_size(kernel_size)
{
    register_parameter(_weight);
    if (_bias) {
        register_parameter(_bias.value());
    }
}

auto ts::naive::Conv2D::create(int in_channels, int out_channels, int kernel_size, int stride, Activation activation,
                               bool use_bias) -> Conv2D
{
    std::vector<int> shape = {kernel_size * kernel_size * in_channels, out_channels};
    Variable<float, 2> weight(std::make_unique<MatrixF>(ts::kaiming_uniform<float, 2>(shape)),
                              std::make_unique<MatrixF>(ts::kaiming_uniform<float, 2>(shape)), "Conv2D(weight)");
    std::optional<Variable<float, 1>> bias = std::nullopt;
    if (use_bias)
        bias = std::make_optional(Variable<float, 1>(std::make_unique<VectorF>(ts::bias_init<float, 1>({out_channels})),
                                                     std::make_unique<VectorF>(ts::bias_init<float, 1>({out_channels})),
                                                     "Conv2D(bias)  "));
    return Conv2D(std::move(weight), std::move(bias), kernel_size, stride, activation);
}

auto ts::naive::Conv2D::operator()(const ts::Tensor<float, 4> &input) -> Tensor<float, 4> { return forward(input); }

auto ts::naive::Conv2D::forward(const ts::Tensor<float, 4> &input) -> ts::Tensor<float, 4>
{
    _input = input;
    auto output = ts::conv_2d(input, _weight.tensor(), _kernel_size, _stride);
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

auto ts::naive::Conv2D::backward(const ts::Tensor<float, 4> &d_output) -> ts::Tensor<float, 4>
{
    auto d_output_(d_output); // cheap, not a deep copy
    if (_activation) {
        d_output_ = _activation.value()->backward(d_output_);
    }
    auto [d_input, d_weight] = ts::conv_2d_backward(_input, _weight.tensor(), d_output_, _kernel_size, _stride);
    _weight.grad() = std::move(d_weight);

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

auto ts::naive::Conv2D::weight() -> ts::Variable<float, 2> & { return _weight; }

auto ts::naive::Conv2D::bias() -> std::optional<std::reference_wrapper<ts::Variable<float, 1>>>
{
    if (_bias.has_value()) {
        return std::make_optional(std::ref(_bias.value()));
    } else {
        return std::nullopt;
    }
}

auto ts::naive::Conv2D::weights() -> VectorRef
{
    std::vector<std::reference_wrapper<ts::GradHolder<float>>> vars;
    vars.emplace_back(std::ref(weight()));
    if (_bias.has_value()) {
        vars.emplace_back(std::ref(bias().value()));
    }
    return vars;
}
