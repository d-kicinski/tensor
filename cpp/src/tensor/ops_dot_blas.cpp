#define OPENBLAS_CONST
#include "ops_dot.hpp"
#include "tensor.hpp"
#include <cblas.h>

namespace ts {


auto outer_product(VectorF const x, VectorF const & y) -> MatrixF
{
    // x or y could be just view on higher dimensional tensor, if I want to use raw pointer to
    // underlining data I have to take that into account
    auto x_data = x.data()->data() + std::distance(x.data().get()->begin(), x.begin());
    auto y_data = y.data()->data() + std::distance(y.data().get()->begin(), y.begin());

    MatrixF result(x.data_size(), y.data_size());
    cblas_sger(CBLAS_ORDER::CblasRowMajor,
               x.data_size(),
               y.data_size(),
               1.0,
               x_data,
               1,
               y_data,
               1,
               result.data()->data(),
               y.data_size());

    return result;
}

auto dot(VectorF const &A, VectorF const &X) -> float
{
    // A or X could be just view on higher dimensional tensor, if I want to use raw pointer to
    // underlining data I have to take that into account
    auto A_data = A.data()->data() + std::distance(A.data().get()->begin(), A.begin());
    auto X_data = X.data()->data() + std::distance(X.data().get()->begin(), X.begin());
    return cblas_sdot(A.data_size(), A_data, 1, X_data, 1);
}

auto dot(MatrixF const &A, VectorF const &X) -> VectorF
{

    // A or X could be just view on higher dimensional tensor, if I want to use raw pointer to
    // underlining data I have to take that into account
    auto A_data = A.data()->data() + std::distance(A.data().get()->begin(), A.begin());
    auto X_data = X.data()->data() + std::distance(X.data().get()->begin(), X.begin());

    VectorF Y(A.shape(0));
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, A.shape(0), A.shape(1),
                1.0f, A_data, A.shape(1), X_data, 1, 0.0f, Y.data()->data(), 1);

    return Y;
}

auto dot(MatrixF const &A, MatrixF const &B, bool A_T, bool B_T) -> MatrixF
{
    int m = A.shape(0);
    int n = B.shape(1);
    int k = A.shape(1);
    int lda = k;
    int ldb = n;
    CBLAS_TRANSPOSE trans_A = CBLAS_TRANSPOSE::CblasNoTrans;
    CBLAS_TRANSPOSE trans_B = CBLAS_TRANSPOSE::CblasNoTrans;

    if (A_T) {
        trans_A = CBLAS_TRANSPOSE::CblasTrans;
        m = A.shape(1);
        k = A.shape(0);
        lda = m;
    }
    if (B_T) {
        trans_B = CBLAS_TRANSPOSE::CblasTrans;
        n = B.shape(0);
        ldb = k;
    }

    // A or B could be just view on higher dimensional tensor, if I want to use raw pointer to
    // underlining data I have to take that into account
    auto A_data = A.data()->data() + std::distance(A.data().get()->begin(), A.begin());
    auto B_data = B.data()->data() + std::distance(B.data().get()->begin(), B.begin());

    MatrixF C(m, n);
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor, trans_A, trans_B, m, n, k,
                1.0f, A_data, lda, B_data, ldb, 0.0f,
                C.data()->data(), C.shape(1));
    return C;
}

auto dot(Tensor<float, 3> const &A, MatrixF const &B) -> Tensor<float, 3>
{
    int batch_size = A.shape(0);
    std::vector<Tensor<float, 2>> partial;
    partial.push_back(dot(A(0), B));
    for (int i = 1; i < batch_size; ++i) {
        partial.push_back(dot(A(i), B));
    }
    return Tensor<float, 3>(partial);
}

} // namespace ts