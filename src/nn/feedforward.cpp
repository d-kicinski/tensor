#include "feedforward.hpp"
namespace ts::nn {


FeedForward::FeedForward(int dim_in, int dim_out, float alpha, bool activation)
    : _alpha(alpha), _activation(activation)
{
    // TODO initialize _weight with random values, like np.random.rand in numpy
   _weights = Matrix(dim_in, dim_out);
   _bias = Matrix(1, dim_out);
}

auto FeedForward::operator()(Matrix inputs) -> Matrix { return forward(inputs); }

auto FeedForward::forward(Matrix inputs) -> Matrix
{
    _x = inputs;
    _y = ts::add(ts::multiply(_x, _weights), _bias);
    if (_activation) {
        _y = ts::maximum(0.0f, _y);  // np.maximum(0, _y)
    }
    return _y;
}

auto FeedForward::backward(Matrix d_y) -> Matrix
{
    if (_activation) {
        d_y = ts::assign_if(d_y, _y <= 0, 0.0f);  // d_y[_y <= 0] = 0;
    }

    _d_weights = ts::multiply(_x, _weights, true);
    _d_bias = ts::sum(d_y, axis=0, keepdims=True);

    auto d_x = ts::multiply(d_y, _weights, false, true);
}

auto FeedForward::update(int step_size) -> void
{
    _d_weights += _weights * _alpha;
    _weights +=  _d_weights * -step_size;
    _bias += _d_bias * -step_size;
}
}
