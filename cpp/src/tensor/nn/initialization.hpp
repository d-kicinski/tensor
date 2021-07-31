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

template <typename Element, int Dim> auto uniform(std::vector<int> const &shape, int fan_out) -> Tensor<Element, Dim>
{
    std::default_random_engine random(69);
    float bound = 1.0f / std::sqrt(fan_out);
    std::uniform_real_distribution<Element> dist{-bound, bound};

    std::array<size_type, Dim> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    std::generate(tensor.begin(), tensor.end(), [&dist, &random]() { return dist(random); });
    return tensor;
}

template <typename Element, int Dim> auto zeros(std::vector<int> const &shape) -> Tensor<Element, Dim>
{
    std::array<size_type, Dim> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    return tensor;
}

template <typename Element, int Dim> auto ones(std::vector<int> const &shape) -> Tensor<Element, Dim>
{
    std::array<size_type, Dim> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    ts::fill_(tensor, Element(1));
    return tensor;
}

template <typename Element, int Dim> auto bernoulli(std::vector<int> const &shape, float p) -> Tensor<Element, Dim>
{
    std::default_random_engine random(69);
    std::bernoulli_distribution dist(p);

    std::array<size_type, Dim> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    std::generate(tensor.begin(), tensor.end(), [&dist, &random]() { return Element(dist(random)); });

    return tensor;
}

} // namespace ts
