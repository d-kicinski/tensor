#include <catch2/catch.hpp>

#include <tensor/nn/layer/batch_normalization.hpp>
#include <tensor/statistics.hpp>

auto random_tensor() -> ts::Tensor<float, 4> {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<float> dist{10.0f};
    ts::Tensor<float, 4> tensor(1, 3, 64, 64);
    std::generate(tensor.begin(), tensor.end(), [&]() { return dist(mt); });
    return tensor;
}

TEST_CASE("batch normalization")
{
    auto input = random_tensor();

    auto layer = ts::BatchNormalization2D(3);
    auto output = layer.forward(input);

   float mean0 = ts::mean(output(0, 0));
   float mean1 = ts::mean(output(0, 1));
   float mean2 = ts::mean(output(0, 2));

   auto d_output = output.clone();
   auto d_input = layer.backward(d_output);

}