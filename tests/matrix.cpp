#define CATCH_CONFIG_MAIN // provides main(); this line is required in only one .cpp file
#include <catch2/catch.hpp>
#include <iostream>

#include "matrix.hpp"

TEST_CASE("Construct float matrix") {
    Matrix<float> matrix(3, 5);
    std::cout << "Hello, from tests!" << std::endl;
}