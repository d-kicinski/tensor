#include <catch2/catch.hpp>
#include <tensor/nn/variable.hpp>

TEST_CASE("create variable via static builder method")
{
    auto variable = ts::Variable<float, 3>::create(3, 2, 5);

    auto expected_shape = std::array<ts::size_type, 3>{3, 2, 5};
    REQUIRE(variable.tensor().shape() == expected_shape);
    REQUIRE(variable.grad().shape() == expected_shape);
}

TEST_CASE("create variable via constructor")
{
    auto variable = ts::Variable<float, 3>({2, 3, 4});
    auto expected_shape = std::array<ts::size_type, 3>{2, 3, 4};
    REQUIRE(variable.tensor().shape() == expected_shape);
    REQUIRE(variable.grad().shape() == expected_shape);
}
