#include "activations.hpp"

template class ts::ActivationBase<float, 2>;
template class ts::ActivationBase<float, 3>;
template class ts::ReLU<float, 2>;
template class ts::ReLU<float, 3>;
template class ts::ActivationFactory<float, 2>;
template class ts::ActivationFactory<float, 3>;
