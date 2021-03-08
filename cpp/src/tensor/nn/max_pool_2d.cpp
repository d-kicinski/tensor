#include "max_pool_2d.hpp"
#include "functional.hpp"

ts::MaxPool2D::MaxPool2D(int kernel_size, int stride) : _kernel_size(kernel_size) , _stride(stride) {}

auto ts::MaxPool2D::create(int kernel_size, int stride) -> MaxPool2D
{
    return MaxPool2D(kernel_size, stride);
}

auto ts::MaxPool2D::operator()(Tensor<float, 4> const & input) -> Tensor<float, 4>
{
    return forward(input);
}

auto ts::MaxPool2D::forward(Tensor<float, 4> const & input) -> Tensor<float, 4>
{
    auto [output, mask] = ts::max_pool_2d(input, _kernel_size, _stride);
    _mask = std::move(mask);
    return std::move(output);
}

auto ts::MaxPool2D::backward(Tensor<float, 4> const & d_output) -> Tensor<float, 4>
{
    return max_pool_2d_backward(d_output, _mask, _kernel_size, _stride);
}