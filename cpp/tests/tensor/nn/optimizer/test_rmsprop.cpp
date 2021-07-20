#include <catch2/catch.hpp>

#include <tensor/nn/optimizer/rmsprop.hpp>

using namespace ts;

TEST_CASE("RMSProp")
{
    auto weight =
        Variable<float, 2>(std::make_unique<Tensor<float, 2>>(Tensor<float, 2>::randn({128, 256})),
                           std::make_unique<Tensor<float, 2>>(Tensor<float, 2>::randn({128, 256})), "Variable");
    std::vector<std::reference_wrapper<GradHolder<float>>> vars;
    vars.emplace_back(std::ref(weight));

    float learning_rate = 1e-3;
    float forgetting_factor = 0.99;
    auto optimizer = RMSProp<float>(learning_rate, vars, forgetting_factor);
    optimizer.step();
    optimizer.step();
    optimizer.step();
}
