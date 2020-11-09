#pragma once

#include <cstddef>

namespace ts {

class Dimensions {

  public:
    using size_type = size_t;
    using dimension_index = int;

    size_type *dimensions;
    size_type data_size;

    ~Dimensions() {
        delete[] dimensions;
    }

    template <typename... Sizes> Dimensions(size_type first, Sizes... rest)
    {
        dimensions = new size_type[sizeof...(rest) + 1];
        set_sizes(0, first, rest...);
    }

    template <typename... Sizes> void set_sizes(dimension_index pos, size_type first, Sizes... rest)
    {
        if (pos == 0) {
            data_size = 1;
        }
        dimensions[pos] = first;
        data_size *= first;

        if constexpr (sizeof...(rest) > 0) {
            set_sizes(pos + 1, rest...);
        }
    }
};

}
