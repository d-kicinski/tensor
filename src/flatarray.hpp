#include <cstddef>

#include "dimensions.hpp"

/**
 *
 * @tparam Element is the type of array element
 * @tparam Dim is the number of dimensions
 * @tparam AllocationFlag is the flag informing whether the dimensions and the elements
 *  are allocated or referenced
 */

template <typename Element, int Dim, bool AllocationFlag = true>
class FlatArray {
  public:
    using size_type = size_t;
    using dimension_index = int;

    template<typename ...Sizes>
    FlatArray(size_type first, Sizes... rest) {
        // allocate memory before set_sizes
       _dimensions = new size_type[Dim];
       set_sizes(0, first, rest...);
       // the data is allocated here, outside of set_sizes
       _data = new Element[_data_size];

       // use default init for all elements
       for (size_type i = 0; i < _data_size; i++) {
           _data[i] = Element();
       }
       _owner = true;
    }

    FlatArray(Dimensions const & sizes, Element* data) {
        _dimensions = new size_type [Dim];
        std::copy(sizes.dimensions, sizes.dimensions + Dim, _dimensions);
        _data_size = sizes.data_size;
        _owner = false;
        _data = data;
    }

  private:
    // sizes of dimensions
    size_type *_dimensions;
    // total number of elements
    size_type _data_size;
    // all the elements; their number is _data_size
    Element * _data;

    bool _owner;

    template <typename... Sizes>
    void set_sizes(dimension_index pos, size_type first, Sizes...rest) {
       if (pos == 0) {
           _data_size = 1;
       }
       _dimensions[pos] = first;
       _data_size *= first;

       if constexpr (sizeof...(rest) > 0) {
           set_sizes(pos+1, rest...);
       }
    }
};