#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>

#include "tensor/data_holder.hpp"

namespace ts {

template <typename Element> auto mean(DataHolder<Element> const &v) -> float
{
    return std::accumulate(v.begin(), v.end(), 0.0f) / std::distance(v.begin(), v.end());
}

template <typename Element> auto max(DataHolder<Element> const &v) -> float
{
    return *std::max_element(v.begin(), v.end());
}

template <typename Element> auto min(DataHolder<Element> const &v) -> float
{
    return *std::min_element(v.begin(), v.end());
}

template <typename Element> auto print_stats(DataHolder<Element> const &v, std::string tag) -> void
{
    std::cerr << tag << ":"
              << " min = " << min(v) << " max = " << max(v) << " mean = " << mean(v) << std::endl;
};

} // namespace ts
