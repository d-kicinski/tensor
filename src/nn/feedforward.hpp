#pragma once

#include <tensor/tensor.hpp>

namespace ts::nn {

class FeedForward {
  public:
    FeedForward();

  private:
    Tensor<float, 2> _weight;
    Tensor<float, 1> _bias;
};

}

