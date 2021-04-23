#define OPENBLAS_CONST
#include "ops_dot_blas.hpp"
#include "tensor.hpp"
#include <cblas.h>

namespace ts::blas {

auto outer_product(VectorF const &x, VectorF const &y) -> MatrixF
{
    // x or y could be just view on higher dimensional tensor, if I want to use raw pointer to
    // underlining data I have to take that into account
    auto x_data = x.data()->data() + std::distance(x.data().get()->begin(), x.begin());
    auto y_data = y.data()->data() + std::distance(y.data().get()->begin(), y.begin());

    MatrixF result(x.data_size(), y.data_size());
    cblas_sger(CBLAS_ORDER::CblasRowMajor, x.data_size(), y.data_size(), 1.0, x_data, 1, y_data, 1,
               result.data()->data(), y.data_size());

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

auto dot(MatrixF const &A, VectorF const &X, bool A_T) -> VectorF
{
    CBLAS_TRANSPOSE trans_A = CBLAS_TRANSPOSE::CblasNoTrans;
    size_type lda = A.shape(1);
    size_type dim_out = A.shape(0);
    if (A_T) {
        trans_A = CBLAS_TRANSPOSE::CblasTrans;
        dim_out = A.shape(1);
    }

    // A or X could be just view on higher dimensional tensor, if I want to use raw pointer to
    // underlining data I have to take that into account
    auto A_data = A.data()->data() + std::distance(A.data().get()->begin(), A.begin());
    auto X_data = X.data()->data() + std::distance(X.data().get()->begin(), X.begin());

    VectorF Y(dim_out);
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor, trans_A, A.shape(0), A.shape(1), 1.0f, A_data, lda, X_data, 1, 0.0f,
                Y.data()->data(), 1);

    return Y;
}

auto dot(MatrixF const &A, MatrixF const &B, bool A_T, bool B_T) -> MatrixF
{
    size_type m = A.shape(0);
    size_type n = B.shape(1);
    if (A_T) {
        m = A.shape(1);
    }
    if (B_T) {
        n = B.shape(0);
    }

    MatrixF C(m, n);
    dot(A, B, C, A_T, B_T);
    return C;
}

auto dot(MatrixF const &A, MatrixF const &B, MatrixF &C, bool A_T, bool B_T, float beta) -> void
{
    size_type m = A.shape(0);
    size_type n = B.shape(1);
    size_type k = A.shape(1);
    size_type lda = k;
    size_type ldb = n;
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

    // A, B or C could be just view on higher dimensional tensor, if I want to use raw pointer to
    // underlining data I have to take that into account
    auto A_data = A.data()->data() + std::distance(A.data().get()->begin(), A.begin());
    auto B_data = B.data()->data() + std::distance(B.data().get()->begin(), B.begin());
    auto C_data = C.data()->data() + std::distance(C.data().get()->begin(), C.begin());

    if (C.shape() != std::array<ts::size_type, 2>{m, n}) {
        return;
    }
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor, trans_A, trans_B, m, n, k, 1.0f, A_data, lda, B_data, ldb, beta, C_data,
                C.shape(1));
}

auto dot(Tensor<float, 3> const &A, MatrixF const &B) -> Tensor<float, 3>
{
    size_type batch_size = A.shape(0);
    std::vector<Tensor<float, 2>> partial;
    partial.push_back(blas::dot(A(0), B));
    for (int i = 1; i < batch_size; ++i) {
        partial.push_back(blas::dot(A(i), B));
    }
    return Tensor<float, 3>(partial);
}

} // namespace ts::blas