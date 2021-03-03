#pragma once
#include <tensor/tensor.hpp>

namespace ts {

template<typename Element>
class SGD
{
  public:
    DataHolder<Element> & _weight;
    float _lr;
    SGD(float lr, DataHolder<Element> & weight) : _lr(lr), _weight(weight) {}

    auto step(DataHolder<Element> const & d_weight) -> void {
        std::transform(_weight.begin(), _weight.end(), d_weight.begin(), _weight.begin(),
                       [&](Element & w, Element & d_w) {
                         return w + (d_w * -_lr);
                       });
    }
};

}
