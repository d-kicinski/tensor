#include "matrix.hpp"
#include <catch2/catch.hpp>

using namespace space;

TEST_CASE("Construct float matrix") { Matrix<float> matrix(3, 5); }

TEST_CASE("construct with list initializer") { Matrix<int> m3{{1, 1}, {1, 1}}; }
