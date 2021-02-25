#include "feed_forward.hpp"

#include <utility>

namespace ts {

FeedForward::FeedForward(int dim_in, int dim_out, FeedForward::OptActivation activation, bool l2, float alpha)
    : _alpha(alpha), _activation(std::move(activation)), _l2(l2)
{
   _weights = ts::MatrixF::randn({dim_in, dim_out});
   _bias = ts::VectorF(dim_out);
}

auto FeedForward::operator()(MatrixF const &inputs) -> MatrixF { return forward(inputs); }

auto FeedForward::forward(MatrixF const &inputs) -> MatrixF
{
    _x = inputs;
    _y = ts::add(ts::dot(_x, _weights), _bias);
    if (_activation) {
        _y = _activation->forward(inputs);
    }
    return _y;
}

auto FeedForward::backward(MatrixF const & d_y) -> MatrixF
{
    auto input(d_y);  // cheap, not a deep copy
    if (_activation) {
        input = _activation->backward(d_y);
    }
    _d_weights = ts::dot(_x, input, true);
    _d_bias = ts::sum(d_y, 0);

    return ts::dot(d_y, _weights, false, true);
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
