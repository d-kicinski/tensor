#include <catch2/catch.hpp>

#include <tensor/nn/optimizer/adagrad.hpp>

using namespace ts;

TEST_CASE("Adagrad")
{
    auto weight =
        Variable<float, 2>(std::make_unique<Tensor<float, 2>>(Tensor<float, 2>::randn({128, 256})),
                           std::make_unique<Tensor<float, 2>>(Tensor<float, 2>::randn({128, 256})), "Variable");
    std::vector<std::reference_wrapper<GradHolder<float>>> vars;
    vars.emplace_back(std::ref(weight));

    float learning_rate = 1e-3;
    auto optimizer = Adagrad<float>(learning_rate, vars);
    optimizer.step();
    optimizer.step();
    optimizer.step();
}
