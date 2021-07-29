#include "lstm_cell.hpp"

#include "tensor/nn/autograd/sigmoid.hpp"
#include "tensor/nn/autograd/tanh.hpp"

ts::LSTMCell::LSTMCell(ts::LSTMCell::Parameters &p) : _p(p) {}

auto ts::LSTMCell::forward(ts::MatrixF const &input, ts::MatrixF const &prev_state_h, ts::MatrixF const &prev_state_c)
    -> ts::MatrixF
{
    _prev_state_c = prev_state_c;
    _x_dim = input.shape(1);

    _xh = ts::concatenate(std::vector<MatrixF>{input, prev_state_h}, 1);

    _state_f = ts::sigmoid(ts::add(ts::dot(_xh, _p.wxf.tensor()), _p.bf.tensor()));
    _state_i = ts::sigmoid(ts::add(ts::dot(_xh, _p.wxi.tensor()), _p.bi.tensor()));
    _state_o = ts::sigmoid(ts::add(ts::dot(_xh, _p.wxo.tensor()), _p.bo.tensor()));

    _state_c_dash = ts::tanh(ts::add(ts::dot(_xh, _p.wxc.tensor()), _p.bc.tensor()));
    _state_c = ts::add(ts::multiply(_state_f, prev_state_c), ts::multiply(_state_i, _state_c_dash));

    // tanh was consciously omitted here
    _state_h = ts::multiply(_state_o, _state_c);
    return _state_h;
}

auto ts::LSTMCell::backward(ts::MatrixF const &d_h_state, ts::MatrixF const &d_c_state)
    -> std::tuple<ts::MatrixF, ts::MatrixF, ts::MatrixF>
{

    auto d_c = ts::add(ts::multiply(_state_o, d_h_state), d_c_state);
    auto d_o = ts::multiply(_state_c, d_h_state);
    auto d_i = ts::multiply(_state_c_dash, d_c);
    auto d_c_dash = ts::multiply(_state_i, d_c);
    auto d_f = ts::multiply(_prev_state_c, d_c);

    auto d_i_input = ts::sigmoid_backward(_state_i, d_i);
    auto d_f_input = ts::sigmoid_backward(_state_f, d_f);
    auto d_o_input = ts::sigmoid_backward(_state_o, d_o);
    auto d_c_dash_input = ts::tanh_backward(_state_c_dash, d_c_dash);

    _p.wxi.grad() += ts::dot(_xh, d_i_input, true, false);
    _p.wxf.grad() += ts::dot(_xh, d_f_input, true, false);
    _p.wxo.grad() += ts::dot(_xh, d_o_input, true, false);
    _p.wxc.grad() += ts::dot(_xh, d_c_dash_input, true, false);
    _p.bi.grad() += ts::sum(d_i_input, 0);
    _p.bf.grad() += ts::sum(d_f_input, 0);
    _p.bo.grad() += ts::sum(d_o_input, 0);
    _p.bc.grad() += ts::sum(d_c_dash_input, 0);

    // [rows, xh] = [rows, i] * [xh, i]
    auto d_xh = ts::dot(d_i_input, _p.wxi.tensor(), false, true);
    d_xh += ts::dot(d_f_input, _p.wxf.tensor(), false, true);
    d_xh += ts::dot(d_o_input, _p.wxo.tensor(), false, true);
    d_xh += ts::dot(d_c_dash_input, _p.wxc.tensor(), false, true);

    auto ret_d_c = ts::multiply(d_c, _state_f);
    auto ret_d_x = ts::slice(d_xh, 0, _x_dim, 1);
    auto ret_d_h = ts::slice(d_xh, _x_dim, d_xh.shape(1), 1);

    return std::make_tuple(ret_d_h, ret_d_c, ret_d_x);
}

auto ts::LSTMCell::state() -> ts::MatrixF & { return _state_h; }

auto ts::LSTMCell::memory() -> ts::MatrixF & { return _state_c; }

auto ts::LSTMCell::loss() -> CrossEntropyLoss & { return _loss_fn; }
