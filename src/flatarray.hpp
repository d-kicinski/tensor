#include <cstddef>
#include <iostream>

#include "dimensions.hpp"
#include "exceptions.hpp"

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
       // btw, calling default constructor is default behaviour of cpp
       for (size_type i = 0; i < _data_size; i++) {
           _data[i] = Element();
       }
       _owner = true;
    }

    FlatArray(Dimensions const & sizes, Element* data) {
        _data = data;
        _dimensions = new size_type[Dim];
        std::copy(sizes.dimensions, sizes.dimensions + Dim, _dimensions);
        _data_size = sizes.data_size;
        _owner = false;
    }

    template <typename ...Indices>
    Element & operator()(size_type first, Indices... rest) {
        return _data[get_index(0, 0, first, rest...)];
    }

    template <typename ...Sizes>
    void resize(Sizes... sizes) {
        if constexpr (AllocationFlag) {
            if (_owner) {
                set_sizes(0, sizes...);
                delete [] _data;
                _data = new Element();
                for (int i = 0; i < _data_size; ++i) {
                    _data[i] = Element();
                }
            } else
                throw space::FlatArrayException();
        } else
            throw space::FlatArrayException();
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

    template<typename... Indices>
    size_type get_index(dimension_index pos, size_type prev_index, size_type first, Indices... rest) {
        size_type index =  (prev_index * _dimensions[pos]) + first;
        if constexpr (sizeof...(rest) > 0) {
            get_index(pos + 1, index, rest...);
        } else {
           return index;
        }
    }

};

template <typename Element, int Dim>
void print_flat_array(FlatArray<Element, Dim> &flat_array)
{
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << "(" << i << "," << j << ")=" << flat_array(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
