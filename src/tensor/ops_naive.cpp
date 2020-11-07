#include "tensor.hpp"
#include "ops_naive.hpp"

namespace ts {

auto multiply(Matrix A, Vector x) -> Vector
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

auto multiply(Matrix A, Matrix B) -> Matrix
{
    // C(m, n) = A(m, k) * B(k, n)
    Matrix C(A.shape()[0], B.shape()[1]);
    int m = A.shape()[0];
    int k = A.shape()[1];
    int n = B.shape()[1];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
    return C;
}

}