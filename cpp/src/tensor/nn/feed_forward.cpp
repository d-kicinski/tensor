#include "feed_forward.hpp"


namespace ts {

FeedForward::FeedForward(Variable<float, 2> weight, Variable<float, 1> bias, Activation activation,
                         bool l2, float alpha)
    : _weight(std::move(weight)), _bias(std::move(bias)), _activation(Activations::get(activation)),
      _alpha(alpha), _l2(l2) {}

auto FeedForward::create(int dim_in, int dim_out, Activation activation, bool l2, float alpha)
-> FeedForward
{
    auto weight =
        Variable<float, 2>(std::make_unique<ts::MatrixF>(ts::MatrixF::randn({dim_in, dim_out})),
                           std::make_unique<ts::MatrixF>(ts::MatrixF::randn({dim_in, dim_out})));

    auto bias = Variable<float, 1>(std::make_unique<ts::VectorF>(ts::VectorF(dim_out)),
                                   std::make_unique<ts::VectorF>(ts::VectorF(dim_out)));
    return FeedForward(std::move(weight), std::move(bias), activation, l2, alpha);
}

auto FeedForward::operator()(MatrixF const &inputs) -> MatrixF { return forward(inputs); }

auto FeedForward::forward(MatrixF const &inputs) -> MatrixF
{
    _x = inputs;
    auto _y = ts::add(ts::dot(_x, _weight.tensor()), _bias.tensor());
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
    _weight.grad() = ts::dot(_x, d_output, true);
    _bias.grad() = ts::sum(d_output, 0);

    return ts::dot(d_output, _weight.tensor(), false, true);
}

auto FeedForward::weight() -> Variable<float, 2> & { return _weight; }

auto FeedForward::bias() -> Variable<float, 1> & { return _bias; }

auto FeedForward::weights() -> VectorRef
{
    std::vector<std::reference_wrapper<ts::GradHolder<float>>> vars;
    vars.emplace_back(std::ref(weight()));
    vars.emplace_back(std::ref(bias()));
    return vars;
}

}
