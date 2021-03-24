#include "ops_common.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cmath>

namespace ts {

// To preserve my sanity:
template auto mask<float, 1>(Tensor<float, 1> const &, std::function<bool (float)>) -> Tensor<bool, 1>;
template auto mask<float, 2>(Tensor<float, 2> const &, std::function<bool (float)>) -> Tensor<bool, 2>;
template auto mask<float, 3>(Tensor<float, 3> const &, std::function<bool (float)>) -> Tensor<bool, 3>;
template auto mask<float, 4>(Tensor<float, 4> const &, std::function<bool (float)>) -> Tensor<bool, 4>;

template auto add_(Tensor<float, 1> const &, Tensor<float, 1> const &) -> void;
template auto add_(Tensor<float, 2> const &, Tensor<float, 2> const &) -> void;
template auto add_(Tensor<float, 3> const &, Tensor<float, 3> const &) -> void;

template auto add(Tensor<float, 1> const &, Tensor<float, 1> const &) -> Tensor<float, 1>;
template auto add(Tensor<float, 2> const &, Tensor<float, 2> const &) -> Tensor<float, 2>;
template auto add(Tensor<float, 3> const &, Tensor<float, 3> const &) -> Tensor<float, 3>;

template auto add(Tensor<int, 1> const &, Tensor<int, 1> const &) -> Tensor<int, 1>;
template auto add(Tensor<int, 2> const &, Tensor<int, 2> const &) -> Tensor<int, 2>;
template auto add(Tensor<int, 3> const &, Tensor<int, 3> const &) -> Tensor<int, 3>;

template auto add(Tensor<float, 2> const &, Tensor<float, 1> const &) -> Tensor<float, 2>;
template auto add(Tensor<float, 3> const &, Tensor<float, 1> const &) -> Tensor<float, 3>;
template auto add(Tensor<int, 2> const &, Tensor<int, 1> const &) -> Tensor<int, 2>;
template auto add(Tensor<int, 3> const &, Tensor<int, 1> const &) -> Tensor<int, 3>;

template auto add_(Tensor<float, 3> const &, Tensor<float, 1> const &) -> void;



template auto maximum(float, Tensor<float, 1> const &) -> Tensor<float, 1>;
template auto maximum(float, Tensor<float, 2> const &) -> Tensor<float, 2>;
template auto maximum(float, Tensor<float, 3> const &) -> Tensor<float, 3>;
template auto maximum(float, Tensor<float, 4> const &) -> Tensor<float, 4>;

template auto log(Tensor<float, 1> const &) -> Tensor<float, 1>;
template auto log(Tensor<float, 2> const &) -> Tensor<float, 2>;
template auto log(Tensor<float, 3> const &) -> Tensor<float, 3>;

template auto exp(Tensor<float, 1> const &) -> Tensor<float, 1>;
template auto exp(Tensor<float, 2> const &) -> Tensor<float, 2>;
template auto exp(Tensor<float, 3> const &) -> Tensor<float, 3>;

template auto pow(Tensor<float, 1> const &, int) -> Tensor<float, 1>;
template auto pow(Tensor<float, 2> const &, int) -> Tensor<float, 2>;
template auto pow(Tensor<float, 3> const &, int) -> Tensor<float, 3>;

template auto sum(Tensor<float, 1> const &) -> float;
template auto sum(Tensor<float, 2> const &) -> float;
template auto sum(Tensor<float, 3> const &) -> float;

template auto assign_if(Tensor<float, 1> const &, Tensor<bool, 1> const &, float) -> Tensor<float, 1>;
template auto assign_if(Tensor<float, 2> const &, Tensor<bool, 2> const &, float) -> Tensor<float, 2>;
template auto assign_if(Tensor<float, 3> const &, Tensor<bool, 3> const &, float) -> Tensor<float, 3>;
template auto assign_if(Tensor<float, 4> const &, Tensor<bool, 4> const &, float) -> Tensor<float, 4>;

template auto apply_if(Tensor<float, 1>, Tensor<bool, 1>, Fn<float>) -> Tensor<float, 1>;
template auto apply_if(Tensor<float, 2>, Tensor<bool, 2>, Fn<float>) -> Tensor<float, 2>;
template auto apply_if(Tensor<float, 3>, Tensor<bool, 3>, Fn<float>) -> Tensor<float, 3>;

template auto apply(Tensor<float, 1> const &, Fn<float>) -> Tensor<float, 1>;
template auto apply(Tensor<float, 2> const &, Fn<float>) -> Tensor<float, 2>;
template auto apply(Tensor<float, 3> const &, Fn<float>) -> Tensor<float, 3>;

template auto multiply(Tensor<float, 1> const & tensor, float value) -> Tensor<float, 1>;
template auto multiply(Tensor<float, 2> const & tensor, float value) -> Tensor<float, 2>;
template auto multiply(Tensor<float, 3> const & tensor, float value) -> Tensor<float, 3>;

template auto multiply(Tensor<float, 1> const &, Tensor<float, 1> const &) -> Tensor<float, 1>;
template auto multiply(Tensor<float, 2> const &, Tensor<float, 2> const &) -> Tensor<float, 2>;

template auto randint(int, int, std::vector<int> const &) -> Tensor<int, 1>;
template auto randint(int, int, std::vector<int> const &) -> Tensor<int, 2>;
template auto randint(int, int, std::vector<int> const &) -> Tensor<int, 3>;

template auto from_vector(std::vector<int>) -> Tensor<int, 1>;
template auto from_vector(std::vector<float>) -> Tensor<float, 1>;

template auto concatenate<int, 1>(std::vector<Tensor<int, 1>> list) -> decltype(auto);
template auto concatenate<float, 1>(std::vector<Tensor<float, 1>> list) -> decltype(auto);

template auto concatenate<int, 2>(std::vector<Tensor<int, 1>> list) -> decltype(auto);
template auto concatenate<float, 2>(std::vector<Tensor<float, 1>> list) -> decltype(auto);

template auto slice(Tensor<float, 1> tensor, int from, int to) -> Tensor<float, 1>;
template auto slice(Tensor<float, 2> tensor, int from, int to) -> Tensor<float, 2>;
template auto slice(Tensor<int, 1> tensor, int from, int to) -> Tensor<int, 1>;
template auto slice(Tensor<int, 2> tensor, int from, int to) -> Tensor<int, 2>;

template auto argmax(Tensor<float, 2> const &) -> Tensor<int, 1>;
template auto argmax(Tensor<int, 2> const &) -> Tensor<int, 1>;


template <typename Element, int Dim>
auto add_(Tensor<Element, Dim> const &x, Tensor<Element, Dim> const &y) -> void
{
    std::transform(x.begin(), x.end(), y.begin(), x.begin(), std::plus<>());
}

template <typename Element, int Dim>
auto add(Tensor<Element, Dim> const &t1, Tensor<Element, Dim> const &t2) -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(t1);
    std::transform(t1.begin(), t1.end(), t2.begin(), result.begin(), std::plus<>());
    return result;
}

template <typename Element>
auto add(Matrix<Element> const &matrix, Vector<Element> const &vector) -> Matrix<Element>
{
    Matrix<Element> result(matrix.shape());
    for (size_type i = 0; i < matrix.shape(0); ++i) {
        Vector<Element> input_row = matrix(i);
        Vector<Element> result_row = result(i);
        std::transform(input_row.begin(), input_row.end(), vector.begin(),
                       result_row.begin(), std::plus());
    }
    return result;
}

template <typename Element>
auto add(Tensor<Element, 3> const &tensor, Vector<Element> const &vector) -> Tensor<Element, 3>
{
    Tensor<Element, 3> result(tensor.shape());
    for (size_type i = 0; i < tensor.shape(0); ++i) {
        for (size_type j = 0; j < tensor.shape(1); ++j) {
            Vector<Element> values = tensor(i, j);
            std::transform(values.begin(), values.end(), vector.begin(),
                           result(i, j).begin(), std::plus());
        }
    }
    return result;
}

template <typename Element>
auto add_(Tensor<Element, 3> const &tensor, Vector<Element> const &vector) -> void
{
    for (size_type i = 0; i < tensor.shape(0); ++i) {
        for (size_type j = 0; j < tensor.shape(1); ++j) {
            Vector<Element> values = tensor(i, j);
            std::transform(values.begin(), values.end(), vector.begin(),
                           values.begin(), std::plus());
        }
    }
}

auto divide(MatrixF const &matrix, VectorF const &vector) -> MatrixF
{
    constexpr float epsilon = 1e-10;
    // TODO: divide(matrix, vector, axis=1)?
    MatrixF result(matrix.shape());
    for (size_type i = 0; i < vector.shape(0); ++i) {
        auto row = matrix(i);
        std::transform(row.begin(), row.end(),
                       result.begin() + (i * row.data_size()),
                       [&vector, &i](float e) { return e / vector(i) + epsilon; });
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
auto assign_if(Tensor<Element, Dim> const &tensor,
               Tensor<bool, Dim> const &predicate,
               Element value) -> Tensor<Element, Dim>
{
    return apply_if(tensor, predicate,
        (std::function<float (float)>) [&](float) { return value; });
}

template <typename Element, int Dim>
auto apply_if(Tensor<Element, Dim> tensor,
              Tensor<bool, Dim> predicate,
              std::function<Element (Element)> fn) -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(tensor.shape());
    std::transform(tensor.begin(), tensor.end(), predicate.begin(), result.begin(),
                   [&](Element & e, bool pred) {
                       return pred ? fn(e) : e;
                   });
    return result;
}

template <typename Element, int Dim>
auto multiply(Tensor<Element, Dim> const &tensor, Element value) -> Tensor<Element, Dim>
{
   auto result = tensor.clone();
   std::transform(tensor.begin(), tensor.end(), result.begin(),
                  [&](Element & e) {
                    return e * value;
                  });
   return result;
}

template <typename Element, int Dim>
auto multiply(Tensor<Element, Dim> const & t1, Tensor<Element, Dim> const & t2) -> Tensor<Element, Dim>
{
    auto result = t1.clone();
    std::transform(t1.begin(), t1.end(),
                   t2.begin(),
                   result.begin(),
                   [&](Element & e1, Element & e2) {
                     return e1 * e2;
                   });
    return result;
}

auto transpose(MatrixF const &matrix) -> MatrixF {
    int m = matrix.shape(1);
    int n = matrix.shape(0);
    MatrixF transposed(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed(i, j) = matrix(j, i);
        }
    }
    return transposed;
}

auto sum(MatrixF const &matrix, int axis) -> VectorF
{
    if (axis == 0) {
        // np.sum(matrix, axis=0, keepdims=True);

        VectorF result(matrix.shape(1));
        for (size_type j = 0; j < matrix.shape(1); ++j) {
            for (size_type i = 0; i < matrix.shape(0); ++i) {
                result(j) += matrix(i, j);
            }
        }
        return result;
    } else if (axis == 1) {
        // np.sum(matrix, axis=1, keepdims=True)

        VectorF result(matrix.shape(0));
        for (size_type i = 0; i < matrix.shape(0); ++i) {
           result(i) = ts::sum(matrix(i));
        }
        return result;
    }
    assert(false);
    return VectorF {};  // silencing gcc warning
}

auto sum_v2(MatrixF const &matrix, int axis) -> VectorF
{
    if (axis == 0) {
        // np.sum(matrix, axis=0, keepdims=True);

        VectorF result(matrix.shape(1));
        for (size_type j = 0; j < matrix.shape(1); ++j) {
            for (size_type i = 0; i < matrix.shape(0); ++i) {
                result(j) += matrix(i, j);
            }
        }
        return result;
    } else if (axis == 1) {
        // np.sum(matrix, axis=1, keepdims=True)

        VectorF result(matrix.shape(0));
        for (size_type i = 0; i < matrix.shape(0); ++i) {
            result(i) = ts::sum(matrix(i));
        }
        return result;
    }
    assert(false);
    return VectorF {};  // silencing gcc warning
}

template<typename Element, int Dim>
auto sum(Tensor<Element, Dim> const & tensor) -> Element
{
    return std::accumulate(tensor.begin(), tensor.end(), Element());
}

auto to_one_hot(Tensor<int, 1> const &vector) -> Tensor<bool, 2>
{
    int max_index = *std::max_element(vector.begin(), vector.end());
    Tensor<bool, 2> one_hot(vector.shape(0), max_index + 1);
    for (size_type i = 0; i < one_hot.shape(0); ++i) {
        one_hot(i, vector(i)) = true;
    }
    return one_hot;
}

auto get(MatrixF const &matrix, Tensor<int, 1> const &indices) -> VectorF
{
    VectorF result(indices.shape());
    for (size_type i = 0; i < matrix.shape(0); ++i) {
        result(i) = matrix(i, indices(i));
    }
    return result;
}

template <typename Element, int Dim>
auto apply(Tensor<Element, Dim> const &tensor, Fn<Element> fn) -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(tensor.shape());
    std::transform(tensor.begin(), tensor.end(), result.begin(), fn);
    return result;
}

template <typename Element, int Dim>
auto log(Tensor<Element, Dim> const &tensor) -> Tensor<Element, Dim>
{
    constexpr float epsilon = 1e-10;
    Fn<float> log = [](Element e){return std::log(e + epsilon); };
    return ts::apply(tensor, log);
}

template <typename Element, int Dim>

auto exp(Tensor<Element, Dim> const &tensor) -> Tensor<Element, Dim>
{
    constexpr float epsilon = 1e-10;
    Fn<float> exp = [](Element e){return std::exp(e) + epsilon; };
    return ts::apply(tensor, exp);
}

template <typename Element, int Dim>
auto pow(Tensor<Element, Dim> const &tensor, int value) -> Tensor<Element, Dim>
{
    Fn<float> pow = [&value](Element e){return std::pow(e, value); };
    return ts::apply(tensor, pow);
}

template <int Dim>
auto randint(int low, int high, std::vector<int> const &shape) -> Tensor<int, Dim>
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(low, high);

    // TODO: this is weird :P
    std::array<size_type , Dim> _shape;
    std::copy(shape.begin(), shape.end(), _shape.begin());
    Tensor<int, Dim> tensor(_shape);
    std::generate(tensor.begin(), tensor.end(), [&]() { return dist(mt); });
    return tensor;
}

template <typename Element>
auto from_vector(std::vector<Element> vector) -> Tensor<Element, 1>
{
    Tensor<Element, 1> array(vector.size());
    std::copy(vector.begin(), vector.end(), array.begin());
    return array;
}

template <typename Element>
auto argmax(Tensor<Element, 2> const &tensor) -> Tensor<int, 1>
{
    int rows = tensor.shape(0);
    Tensor<int, 1> indexes(rows);
    for (int i = 0; i < rows; ++i) {
        Tensor<Element, 1> row = tensor(i);
        int index = std::max_element(row.begin(), row.end()) - row.begin();
        indexes[i] = index;
    }
    return indexes;
}

}
