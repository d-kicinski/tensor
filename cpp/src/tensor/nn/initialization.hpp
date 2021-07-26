#pragma once
#include <tensor/tensor.hpp>

namespace ts {

template <typename Element, int Dim> auto kaiming_uniform(std::vector<int> const &shape) -> Tensor<Element, Dim>
{
    std::default_random_engine random(69);
    float fan_in = shape[0];

    float gain = (float)(M_SQRT2) / std::sqrt(fan_in);
    float std = gain / std::sqrt(fan_in);
    float bound = std::sqrt(3.0f) * std;
    std::uniform_real_distribution<Element> dist{-bound, bound};

    std::array<size_type, Dim> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    std::generate(tensor.begin(), tensor.end(), [&dist, &random]() { return dist(random); });
    return tensor;
}

template <typename Element, int Dim>
auto standard_normal(std::vector<int> const &shape, Element scale) -> Tensor<Element, Dim>
{
    std::default_random_engine random(69);
    std::normal_distribution<Element> dist;

    std::array<size_type, Dim> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    std::generate(tensor.begin(), tensor.end(), [&dist, &random, &scale]() { return scale * dist(random); });
    return tensor;
}

template <typename Element, int Dim> auto bias_init(std::vector<int> const &shape) -> Tensor<Element, Dim>
{
    std::default_random_engine random(69);
    float fan_in = shape[0];

    float bound = 1.0f / std::sqrt(fan_in);
    std::uniform_real_distribution<Element> dist{-bound, bound};

    std::array<size_type, Dim> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    std::generate(tensor.begin(), tensor.end(), [&dist, &random]() { return dist(random); });
    return tensor;
}

} // namespace ts
