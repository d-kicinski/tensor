#include "feed_forward.hpp"
#include "tensor/nn/initialization.hpp"

namespace ts {

FeedForward::FeedForward(Variable<float, 2> weight, std::optional<Variable<float, 1>> bias, Activation activation)
    : _weight(std::move(weight)), _bias(std::move(bias)), _activation(Activations::get(activation)),
      _use_bias(_bias.has_value())
{
    register_parameters(_weight);
    if (_bias.has_value()) {
        register_parameters(_bias.value());
    }
}

FeedForward::FeedForward(int dim_in, int dim_out, Activation activation, bool use_bias)
    : _weight(std::make_unique<ts::MatrixF>(ts::kaiming_uniform<float, 2>({dim_in, dim_out})),
              std::make_unique<ts::MatrixF>(ts::zeros<float, 2>({dim_in, dim_out})), "FeedForward(weight)"),
      _bias(std::nullopt), _activation(Activations::get(activation)), _use_bias(use_bias)
{
    register_parameters(_weight);
    if (use_bias) {
        _bias = std::make_optional(
            ts::Variable<float, 1>(std::make_unique<ts::VectorF>(ts::uniform<float, 1>({dim_out}, dim_out)),
                                   std::make_unique<ts::VectorF>(ts::zeros<float, 1>({dim_out})), "FeedForward(bias)"));
        register_parameters(_bias.value());
    }
}

auto FeedForward::create(int dim_in, int dim_out, Activation activation, bool use_bias) -> FeedForward
{
    auto weight = Variable<float, 2>(std::make_unique<ts::MatrixF>(ts::kaiming_uniform<float, 2>({dim_in, dim_out})),
                                     std::make_unique<ts::MatrixF>(ts::zeros<float, 2>({dim_in, dim_out})),
                                     "FeedForward(weight)");
    std::optional<Variable<float, 1>> bias = std::nullopt;
    if (use_bias) {
        bias = std::make_optional(Variable<float, 1>(std::make_unique<ts::VectorF>(ts::uniform<float, 1>({dim_out}, dim_out)),
                                                     std::make_unique<ts::VectorF>(ts::zeros<float, 1>({dim_out})),
                                                     "FeedForward(bias)"));
    }
    return FeedForward(std::move(weight), std::move(bias), activation);
}

auto FeedForward::operator()(MatrixF const &inputs) -> MatrixF { return forward(inputs); }

auto FeedForward::forward(MatrixF const &inputs) -> MatrixF
{
    _x = inputs.clone();
    auto _y = ts::dot(_x, _weight.tensor());
    if (_use_bias) {
        _y = ts::add(_y, _bias.value().tensor());
    }

    if (_activation) {
        _y = _activation.value()->forward(_y);
    }
    return _y;
}

auto FeedForward::backward(MatrixF const &d_y) -> MatrixF
{
    MatrixF d_output = d_y.clone();
    if (_activation) {
        d_output = _activation.value()->backward(d_output);
    }
    _weight.grad() += ts::dot(_x, d_output, true);

    if (_use_bias) {
        _bias.value().grad() += ts::sum(d_output, 0);
    }
//    auto [min_x, max_x] = std::minmax_element(_x.begin(), _x.end());
//    auto [min_y, max_y] = std::minmax_element(d_y.begin(), d_y.end());
//    auto [min_g, max_g] = std::minmax_element(_weight.grad().begin(), _weight.grad().end());

    return ts::dot(d_output, _weight.tensor(), false, true);
}

auto FeedForward::weight() -> Variable<float, 2> & { return _weight; }

auto FeedForward::bias() -> std::optional<std::reference_wrapper<Variable<float, 1>>>
{
    if (_bias.has_value()) {
        return std::make_optional(std::ref(_bias.value()));
    } else {
        return std::nullopt;
    }
}

auto FeedForward::weights() -> VectorRef
{
    std::vector<std::reference_wrapper<ts::GradHolder<float>>> vars;
    vars.emplace_back(std::ref(weight()));
    if (_use_bias) {
        vars.emplace_back(std::ref(bias().value()));
    }
    return vars;
}

} // namespace ts
