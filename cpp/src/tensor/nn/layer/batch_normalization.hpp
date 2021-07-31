#pragma once

#include "tensor/nn/initialization.hpp"
#include "tensor/nn/parameters_registry.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {
class BatchNormalization2D : public ParameterRegistry<float> {
  public:
    BatchNormalization2D(int channels_in)
        : _gamma(Variable<float, 1>::create(ts::ones<float, 1>({channels_in}))),
          _bias(Variable<float, 1>::create(ts::zeros<float, 1>({channels_in})))
    {
        register_parameters(_gamma);
        register_parameters(_bias);
    }

    auto forward(Tensor<float, 4> input) -> Tensor<float, 4>
    {
        // input: [batch_size, channel_in, height, width]
        auto [B, C, H, W] = input.shape();

        // compute statistics along C axis
        VectorF mean(C);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                mean.at(c) += ts::sum(input(b, c));
            }
        }
        ts::multiply(mean, static_cast<float>(1.0 / (B * H * W)));

        _var = VectorF(C);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                _var(c) +=
                    ts::sum(ts::apply<float>(input(b, c), [&](float const &e) { return std::pow(e - mean.at(c), 2); }));
            }
        }
        ts::multiply(_var, static_cast<float>(1.0 / (B * H * W)));

        _update_running_variables(mean, _var);

        _stddev = ts::apply<float>(_var, [this](auto const &e) { return std::sqrt(e + _epsilon); });

        _input_centered = input.clone();
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                ts::apply_<float>(_input_centered(b, c), [&](float const &e) { return e - mean.at(c); });
            }
        }

        _input_normalized = _input_centered.clone();
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                ts::apply_<float>(_input_normalized(b, c),
                                  [&, this](float const &e) { return e / (_stddev.at(c) + _epsilon); });
            }
        }

        auto input_scaled = _input_normalized.clone();
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                ts::apply_<float>(input_scaled(b, c), [&, this](float const &e) {
                    return _gamma.tensor().at(c) * e + _bias.tensor().at(c);
                });
            }
        }
        return input_scaled;
    }

    auto backward(Tensor<float, 4> const &d_output) -> Tensor<float, 4>
    {
        auto [B, C, H, W] = d_output.shape();
        _bias.grad() += _sum_channel_wise(d_output);
        _gamma.grad() += _sum_channel_wise(ts::multiply(_input_normalized, d_output));

        auto d_normalized = Tensor<float, 4>(d_output.shape());
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                ts::apply_<float>(d_normalized(b, c), [&, this](float const &e) { return _gamma.tensor().at(c) * e; });
            }
        }

        auto d_var = VectorF(_var.shape());
        auto pow_var = ts::pow(_var, -3.0f / 2.0f);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                d_var(c) += ts::sum(
                    ts::apply<float>(d_normalized(b, c), _input_centered(b, c),
                                     [&](float const &e1, float const &e2) { return e1 * e2 * -0.5 * pow_var.at(c); }));
            }
        }
        ts::multiply(d_var, static_cast<float>(1.0 / (B * H * W)));

        int batch_size = B;
        auto aux_input_centered =
            ts::apply<float>(_input_centered, [batch_size](auto const &e) { return 2 * e / batch_size; });
        auto stddev_inv = ts::apply<float>(_stddev, [](auto const &e) { return 1.0f / e; });

        auto d_mean = VectorF(_var.shape());
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                d_mean(c) += ts::sum(ts::apply<float>(
                    d_normalized(b, c), aux_input_centered(b, c),
                    [&](float const &e1, float const &e2) { return e1 * -stddev_inv.at(c) + -e2 * d_var.at(c); }));
            }
        }

        auto d_input = Tensor<float, 4>(d_output.shape());
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        d_input.at({b, c, h, w}) = d_normalized.at({b, c, h, w}) * stddev_inv.at(c) +
                                                   d_var.at(c) * aux_input_centered.at({b, c, h, w}) +
                                                   d_mean.at(c) / batch_size;
                    }
                }
            }
        }
        return d_input;
    }

    auto _sum_channel_wise(Tensor<float, 4> const &input) -> VectorF
    {
        auto [B, C, H, W] = input.shape();
        auto result = VectorF(C);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                result.at(c) += ts::sum(input(b, c));
            }
        }
        return result;
    }

    auto _update_running_variables(VectorF const &mean, VectorF const &var) -> void
    {
        if (_running_mean.data_size() == 0 or _running_var.data_size() == 0) {
            _running_mean = mean.clone();
            _running_var = var.clone();
        }
        for (int i = 0; i < mean.data_size(); ++i) {
            _running_mean.at(i) =
                _gamma.tensor().at(i) * _running_mean.at(i) + (1.0f - _gamma.tensor().at(i)) * mean.at(i);
            _running_var.at(i) =
                _gamma.tensor().at(i) * _running_var.at(i) + (1.0f - _gamma.tensor().at(i)) * var.at(i);
        }
    }

  private:
    Variable<float, 1> _gamma;
    Variable<float, 1> _bias;

    VectorF _running_mean{};
    VectorF _running_var{};
    VectorF _stddev{};
    Tensor<float, 4> _input_centered{};
    Tensor<float, 4> _input_normalized{};
    VectorF _var{};

    float _epsilon = 1e-8;
};
} // namespace ts