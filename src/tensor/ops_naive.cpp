#include "ops_naive.hpp"
#include "tensor.hpp"

namespace ts {

auto multiply(Matrix const &A, Vector const &x) -> Vector
{
    // C(m) = A(m, k) * x(k)
    Vector y(A.shape()[0]);
    int m = A.shape()[0];
    int k = A.shape()[1];

    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            float value = A(i, p) * x(p);
            y(i) += value;
        }
    }
    return y;
}

auto multiply(Matrix const &A, Matrix const &B, bool A_T, bool B_T) -> Matrix
{
    // C(m, n) = A(m, k) * B(k, n)
    int m = A_T ? A.shape()[1] : A.shape()[0];
    int n = B_T ? B.shape()[0]: B.shape()[1];
    int k = A_T ? A.shape()[0] : A.shape()[1];

    Matrix C(m, n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                if (A_T) {
                    C(i, j) += A(p, i) * B(p, j);
                } else if (B_T) {
                    C(i, j) += A(i, p) * B(j, p);
                } else {
                    C(i, j) += A(i, p) * B(p, j);
                }
            }
        }
    }
    return C;
}

auto multiply(Tensor<float, 3> const &A, Matrix const &B) -> Tensor<float, 3>
{
    int const batch_size = A.dimensions()[0];
    std::vector<Matrix> partial;
    partial.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        Tensor<float, 2> matrix = A(i);
        Tensor<float, 2> sub_result = multiply(matrix, B);
        partial.push_back(sub_result);
    }
    return Tensor<float, 3>(partial);
}

auto transpose(const Matrix & matrix) -> Matrix {
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

} // namespace ts