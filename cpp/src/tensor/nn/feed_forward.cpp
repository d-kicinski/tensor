#include "feed_forward.hpp"


namespace ts {

FeedForward::FeedForward(int dim_in, int dim_out, Activation activation, bool l2, float alpha)
    : _alpha(alpha), _l2(l2)
{
   _weights = ts::MatrixF::randn({dim_in, dim_out});
   _bias = ts::VectorF(dim_out);
   _activation = Activations::get(activation);
}

auto FeedForward::operator()(MatrixF const &inputs) -> MatrixF { return forward(inputs); }

auto FeedForward::forward(MatrixF const &inputs) -> MatrixF
{
    _x = inputs;
    auto _y = ts::add(ts::dot(_x, _weights), _bias);
    if (_activation) {
        _y = _activation.value()->forward(_y);
    }
    return _y;
}

auto FeedForward::backward(MatrixF const & d_y) -> MatrixF
{
    MatrixF d_output = d_y.clone() ;
    if (_activation) {
        d_output = _activation.value()->backward(d_output);
    }
    _d_weights = ts::dot(_x, d_output, true);
    _d_bias = ts::sum(d_output, 0);

    return ts::dot(d_output, _weights, false, true);
}

auto FeedForward::update(float step_size) -> void
{
    if (_l2) {
        _weights += ts::multiply(_weights, -_alpha);
    }
    _weights +=  ts::multiply(_d_weights, -step_size);
    _bias += ts::multiply(_d_bias, -step_size);
}

auto FeedForward::weight() -> MatrixF { return _weights; }

auto FeedForward::bias() -> VectorF { return _bias; }

}
