#pragma once

namespace ts {

template <typename Element> auto print_stats(DataHolder<Element> const &v, std::string tag) -> void
{
    std::cerr << tag << ":"
              << " min = " << *std::min_element(v.begin(), v.end())
              << " max = " << *std::max_element(v.begin(), v.end())
              << " mean = " << std::accumulate(v.begin(), v.end(), 0.0f) / std::distance(v.begin(), v.end())
              << std::endl;
};

} // namespace
