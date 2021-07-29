#pragma once
#pragma once

#include "tensor/nn/cross_entropy_loss.hpp"
#include "tensor/nn/variable.hpp"

namespace ts {

class LSTMCell {
  public:
    struct Parameters {
        Variable<float, 2> wxf;
        Variable<float, 2> wxi;
        Variable<float, 2> wxo;
        Variable<float, 2> wxc;
        Variable<float, 1> bf;
        Variable<float, 1> bi;
        Variable<float, 1> bo;
        Variable<float, 1> bc;
    };

    explicit LSTMCell(Parameters &p);

    auto forward(ts::MatrixF const &input, ts::MatrixF const &prev_state_h, ts::MatrixF const & prev_state_c) -> ts::MatrixF;

    auto backward(ts::MatrixF const &d_h_state, ts::MatrixF const &d_c_state) -> std::tuple<ts::MatrixF, ts::MatrixF, ts::MatrixF>;

    auto state() -> ts::MatrixF &;

    auto memory() -> ts::MatrixF &;

    auto loss() -> CrossEntropyLoss &;


  private:
    Parameters &_p;

    CrossEntropyLoss _loss_fn{};

    int _x_dim;
    MatrixF _xh{};
    MatrixF _prev_state_c{};
    MatrixF _state_i{};
    MatrixF _state_o{};
    MatrixF _state_f{};
    MatrixF _state_c{};
    MatrixF _state_c_dash{};
    MatrixF _state_h{};
};
} // namespace ts
