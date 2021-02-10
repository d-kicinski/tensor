#include <catch2/catch.hpp>
#include "tensor/nn/softmax.hpp"


TEST_CASE("softmax")
{
   ts::MatrixF log_probabilities = ts::MatrixF::randn({32, 3});
   ts::MatrixF probabilities = ts::softmax(log_probabilities);

   for (int i = 0; i < probabilities.shape(0); ++i) {
        auto row = probabilities(i);
        REQUIRE(Approx(ts::sum(row)) == 1);
   }
}
