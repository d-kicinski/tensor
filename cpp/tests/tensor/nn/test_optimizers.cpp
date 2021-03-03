#include <catch2/catch.hpp>
#include <tensor/nn/optimizers.hpp>

using namespace ts;

TEST_CASE("SGD") {
    auto weight = Tensor<float, 2>::randn({128, 256});
    float learning_rate = 1e-3;
    auto optimizer = SGD<float>(learning_rate, weight);

    auto d_weight = Tensor<float, 2>::randn({128, 256});
    optimizer.step(d_weight);
}