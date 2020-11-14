#include "ops_common.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>

namespace ts {

// To preserve my sanity:
template auto mask<float, 1>(Tensor<float, 1> const &, std::function<bool (float)>) -> Tensor<bool, 1> ;
template auto mask<float, 2>(Tensor<float, 2> const &, std::function<bool (float)>) -> Tensor<bool, 2> ;
template auto mask<float, 3>(Tensor<float, 3> const &, std::function<bool (float)>) -> Tensor<bool, 3> ;

template auto add(Tensor<float, 1> const &, Tensor<float, 1> const &) -> Tensor<float, 1>;
template auto add(Tensor<float, 2> const &, Tensor<float, 2> const &) -> Tensor<float, 2>;
template auto add(Tensor<float, 3> const &, Tensor<float, 3> const &) -> Tensor<float, 3>;

template auto maximum(float, Tensor<float, 1> const &) -> Tensor<float, 1>;
template auto maximum(float, Tensor<float, 2> const &) -> Tensor<float, 2>;
template auto maximum(float, Tensor<float, 3> const &) -> Tensor<float, 3>;

template auto assign_if(Tensor<float, 1> const &, Tensor<bool, 1> const &, float) -> Tensor<float, 1>;
template auto assign_if(Tensor<float, 2> const &, Tensor<bool, 2> const &, float) -> Tensor<float, 2>;
template auto assign_if(Tensor<float, 3> const &, Tensor<bool, 3> const &, float) -> Tensor<float, 3>;

template auto multiply(Tensor<float, 1> const & tensor, float value) -> Tensor<float, 1>;
template auto multiply(Tensor<float, 2> const & tensor, float value) -> Tensor<float, 2>;
template auto multiply(Tensor<float, 3> const & tensor, float value) -> Tensor<float, 3>;


template <typename Element, int Dim>
auto add(Tensor<Element, Dim> const &t1, Tensor<Element, Dim> const &t2) -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(t1);
    std::transform(t1.begin(), t1.end(), t2.begin(), result.begin(), std::plus<>());
    return result;
}

auto add(Matrix const &matrix, Vector const &vector) -> Matrix
{
    Matrix result(matrix.shape());
    for (int i = 0; i < matrix.shape()[0]; ++i) {
        auto column = matrix(i);
        std::transform(column.begin(), column.end(), vector.begin(),
                       result.begin() + (i * column.data_size()), std::plus());
    }
    return result;
}

template <typename Element, int Dim>
auto maximum(Element value, Tensor<Element, Dim> const &tensor) -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(tensor.shape());
    std::transform(tensor.begin(), tensor.end(), result.begin(),
                   [&](Element & e) {
                     return e < value ? value : e;
                   });
    return result;
}

template <typename Element, int Dim>
auto mask(Tensor<Element, Dim> const &tensor, std::function<bool(Element)> fn) -> Tensor<bool, Dim>
{
    Tensor<bool, Dim> mask(tensor.shape());
    std::transform(tensor.begin(), tensor.end(), mask.begin(), fn );
    return mask;
}

template <typename Element, int Dim>
auto assign_if(Tensor<Element, Dim> const &tensor, Tensor<bool, Dim> const &predicate, Element value)
    -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(tensor.shape());
    std::transform(tensor.begin(), tensor.end(), predicate.begin(), result.begin(),
                   [&](Element & e, bool pred) {
                      return pred ? value : e;
                   });
    return result;
}

template <typename Element, int Dim>
auto multiply(Tensor<Element, Dim> const &tensor, Element value) -> Tensor<Element, Dim>
{
   auto result(tensor);
   std::transform(tensor.begin(), tensor.end(), result.begin(),
                  [&](Element & e) {
                    return e * value;
                  });
   return result;
}

auto transpose(Matrix const &matrix) -> Matrix {
    int m = matrix.shape()[1];
    int n = matrix.shape()[0];
    Matrix transposed(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed(i, j) = matrix(j, i);
        }
    }
    return transposed;
}

auto sum(Matrix const &matrix, int axis) -> Vector
{
    // ts::sum(d_y, axis=0, keepdims=True);
    assert(axis == 0);
    Vector result(matrix.shape()[1]);
    for (int j = 0; j < matrix.shape()[1]; ++j) {
        for (int i = 0; i < matrix.shape()[0]; ++i) {
            auto val = matrix(i, j);
            result(j) += matrix(i, j);
        }
    }
    return result;
}

}
