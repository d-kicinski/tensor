#include <catch2/catch.hpp>

#include <tensor/nn/optimizer/adam.hpp>

using namespace ts;

TEST_CASE("Adam")
{
    auto weight =
        Variable<float, 2>(std::make_unique<Tensor<float, 2>>(Tensor<float, 2>::randn({128, 256})),
                           std::make_unique<Tensor<float, 2>>(Tensor<float, 2>::randn({128, 256})), "Variable");
    std::vector<std::reference_wrapper<GradHolder<float>>> vars;
    vars.emplace_back(std::ref(weight));

    auto optimizer = Adam<float>(vars);
    optimizer.step();
    optimizer.step();
    optimizer.step();
}
