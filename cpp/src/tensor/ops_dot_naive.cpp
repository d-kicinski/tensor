#include "ops_dot_naive.hpp"
#include "tensor.hpp"

namespace ts::naive {

auto outer_product(VectorF const &x, VectorF const &y) -> MatrixF
{
    MatrixF result(x.data_size(), y.data_size());
    for (size_type i = 0; i < x.data_size(); ++i) {
        for (size_type j = 0; j < y.data_size(); ++j) {
            result(i, j) = x(i) * y(j);
        }
    }
    return result;
}

auto dot(VectorF const &a, VectorF const &b) -> float
{
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

auto dot(MatrixF const &A, VectorF const &x, bool A_T) -> VectorF
{
    // C(m) = A(m, k) * x(k)
    int m = A_T ? A.shape(1) : A.shape(0);
    int k = A_T ? A.shape(0) : A.shape(1);

    VectorF y(m);

    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            float value;
            if (A_T)
                value = A(p, i) * x(p);
            else
                value = A(i, p) * x(p);
            y(i) += value;
        }
    }
    return y;
}

auto dot(MatrixF const &A, MatrixF const &B, bool A_T, bool B_T) -> MatrixF
{
    // C(m, n) = A(m, k) * B(k, n)
    int m = A_T ? A.shape(1) : A.shape(0);
    int n = B_T ? B.shape(0) : B.shape(1);

    MatrixF C(m, n);
    dot(A, B, C, A_T, B_T);
    return C;
}


auto dot(MatrixF const &A, MatrixF const &B, MatrixF &C, bool A_T, bool B_T, float beta) -> void
{
    // C(m, n) = A(m, k) * B(k, n) + beta * C(m, n)
    int m = A_T ? A.shape(1) : A.shape(0);
    int n = B_T ? B.shape(0) : B.shape(1);
    int k = A_T ? A.shape(0) : A.shape(1);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int p = 0; p < k; p++) {
                if (A_T) {
                    acc += A(p, i) * B(p, j);
                } else if (B_T) {
                    acc += A(i, p) * B(j, p);
                } else {
                    acc += A(i, p) * B(p, j);
                }
            }
            C(i, j) = acc + beta * C(i, j);
        }
    }
}

auto dot(Tensor<float, 3> const &A, MatrixF const &B) -> Tensor<float, 3>
{
    int batch_size = A.shape(0);
    std::vector<Tensor<float, 2>> partial;
    partial.push_back(naive::dot(A(0), B));
    for (int i = 1; i < batch_size; ++i) {
        partial.push_back(naive::dot(A(i), B));
    }
    return Tensor<float, 3>(partial);
}

} // namespace ts::naive