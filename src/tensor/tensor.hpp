#pragma once

#include <cstddef>
#include <iostream>
#include <sstream>

#include "dimensions.hpp"
#include "iterator.hpp"
#include "ops.hpp"
#include "exceptions.hpp"

#include "tensor_forward.hpp"

namespace ts {


/**
 *
 * @tparam Element is the type of array element
 * @tparam Dim is the number of dimensions
 * @tparam AllocationFlag is the flag informing whether the dimensions and the elements
 *  are allocated or referenced
 */

template <typename Element, int Dim, bool AllocationFlag> class Tensor {

  public:
    using size_type = size_t;
    using dimension_index = int;

    auto data_size() const -> size_type { return _data_size; }
    auto data() const -> Element * { return _data; };
    auto owner() const -> bool { return _owner; };
    auto dimensions() const -> size_type * { return _dimensions; };

    Tensor()
    {
        _dimensions = new size_type[Dim];
        _data_size = 0;
        _data = nullptr;
    }

    ~Tensor()
    {
        // TODO: this causes SEGFAULT :/
        //        if constexpr (AllocationFlag) {
        //            delete[] _dimensions;
        //            if (_owner) {
        //                delete _data;
        //            }
        //        }
    }

    Tensor(std::initializer_list<Element> list)
    {
        _dimensions = new size_type[Dim];
        _data_size = 0;
        _data = nullptr;

        if (list.size() == 0) {
            return;
        }
        _data_size = list.size();
        _dimensions[0] = _data_size;
        _data = new Element[_data_size];
        size_type i = 0;
        for (Element const &element : list) {
            _data[i++] = element;
        }
    }

    Tensor(std::initializer_list<Tensor<Element, Dim - 1>> list)
    {
        _dimensions = new size_type[Dim];
        if (list.size() == 0) {
            _data_size = 0;
            _data = nullptr;
            return;
        }

        size_type first_dim = list.size();
        size_type counter = 0;
        for (Tensor<Element, Dim - 1> const &e : list) {
            if (counter == 0) {
                _dimensions[0] = first_dim;
                std::copy(e.dimensions(), e.dimensions() + Dim - 1, _dimensions + 1);
            }
            counter += e.data_size();
        }
        _data_size = counter;
        _data = new Element[_data_size];

        Element *data_end = _data;
        for (Tensor<Element, Dim - 1> const &e : list) {
            data_end = std::copy(e._data, e._data + e._data_size, data_end);
        }
    }

    template <typename... Sizes> Tensor(size_type first, Sizes... rest)
    {
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

    Tensor(Dimensions const &sizes, Element *data)
    {
        _data = data;
        _dimensions = new size_type[Dim];
        std::copy(sizes.dimensions, sizes.dimensions + Dim, _dimensions);
        _data_size = sizes.data_size;
        _owner = false;
    }

    Tensor(Tensor const & x)
    {
        _data_size = x.data_size();
        _data = new Element[x.data_size()];
        _dimensions = new size_type[Dim];
        std::copy(x.data(), x.data() + data_size(), _data);
        std::copy(x.dimensions(), x.dimensions() + Dim, _dimensions);
    }

    // getting a reference to a subarray
    template <bool AllocationFlag2>
    Tensor(Tensor<Element, Dim + 1, AllocationFlag2> array, size_type index)
    {
        _data_size = array.data_size() / array.dimensions()[0];
        _dimensions = array.dimensions() + 1;
        size_type m = 1;
        for (dimension_index i = 0; i < Dim; ++i) {
            m *= _dimensions[i];
        }
        _data = array.data() + index * m;
        _owner = false;
    }

    using iterator = IteratorClass<Element>;
    auto begin() -> iterator { return iterator::begin(data(), data_size()); }
    auto end() -> iterator { return iterator::end(data(), data_size()); }

    using const_iterator = IteratorClass<Element>;
    auto begin() const -> const_iterator { return iterator::begin(data(), data_size()); }
    auto end() const -> const_iterator { return iterator::end(data(), data_size()); }

    auto shape() -> std::vector<int>
    {
        std::vector<int> shape(_dimensions, _dimensions + Dim);
        shape.resize(Dim);
        return shape;
    }

    template <typename... Indices>
    auto operator()(size_type first, Indices... rest) -> decltype(auto)
    {
        if constexpr (sizeof...(Indices) == Dim - 1) {
            return _data[get_index(0, 0, first, rest...)];
        } else if constexpr (Dim >= 2 && sizeof...(Indices) == 0) {
            return Tensor<Element, Dim - 1, false>(*this, first);
        } else if constexpr (Dim >= 2 && sizeof...(Indices) < Dim - 1) {
            return Tensor<Element, Dim - 1, false>(*this, first)(rest...);
        } else {
            throw TensorException("operator()");
        }
    }

    auto operator[](size_type i) -> decltype(auto)
    {
        if constexpr (Dim == 1) {
            return _data[i];
        } else {
            return Tensor<Element, Dim - 1, false>(*this, i);
        }
    }

    auto operator[](size_type i) const -> decltype(auto)
    {
        if constexpr (Dim == 1) {
            return _data[i];
        } else {
            return Tensor<Element, Dim - 1, false>(*this, i);
        }
    }

    auto operator==(Tensor<Element, Dim, AllocationFlag> other) const -> bool
    {
        if (_data_size != other.data_size()) {
            return false;
        }

        for (int i = 0; i < _data_size; ++i) {
            if (other.data()[i] != _data[i]) {
                return false;
            }
        }
        return true;
    }

    auto operator!=(Tensor<Element, Dim, AllocationFlag> const &other) -> bool
    {
        return !(*this == other);
    }

    auto operator=(Tensor const &x) -> Tensor &
    {
        if constexpr (AllocationFlag) {
            if (_data_size != x.data_size()) {
                if (!_owner) {
                    std::ostringstream ss;
                    ss << "operator= NON-OWNER CANNOT BE RESIZED. size1: " << _data_size
                       << " size2: " << x.data_size();
                    throw TensorException(ss.str());
                }
                delete[] _data;
                _data_size = x.data_size();
                _data = new Element[data_size()];
            }

            std::copy(x.data(), x.data() + _data_size, _data);
            std::copy(x.dimensions(), x.dimensions() + Dim, _dimensions);
        } else {
            _dimensions = x.dimensions();
            _data_size = x.data_size();
            _data = x.data();
        }
        return *this;
    }

    template <bool copy2> auto operator=(Tensor<Element, Dim, copy2> const & x) -> Tensor &
    {
        if constexpr (AllocationFlag) {
            if (_data_size != x.data_size()) {
                if (!_owner) {
                    std::ostringstream ss;
                    ss << "operator= NON-OWNER CANNOT BE RESIZED. size1: " << _data_size
                       << " size2: " << x.data_size();
                    throw TensorException(ss.str());
                }
                delete[] _data;
                _data_size = x.data_size();
                _data = new Element[_data_size];
            }

            std::copy(x.data(), x.data() + _data_size, _data);
            std::copy(x.dimensions(), x.dimensions() + Dim, _dimensions);
        } else {
            _dimensions = x.dimensions();
            _data_size = x.data_size();
            _data = x.data();
        }
        return *this;
    }

    template <typename... Sizes> auto resize(Sizes... sizes) -> void
    {
        if constexpr (AllocationFlag) {
            if (_owner) {
                set_sizes(0, sizes...);
                delete[] _data;
                _data = new Element();
                for (int i = 0; i < _data_size; ++i) {
                    _data[i] = Element();
                }
            } else
                throw TensorException();
        } else
            throw TensorException();
    }

  private:
    // sizes of dimensions
    size_type *_dimensions;
    // total number of elements
    size_type _data_size;
    // all the elements; their number is _data_size
    Element *_data;

    bool _owner = true;

    friend class Tensor<Element, Dim - 1, true>;
    friend class Tensor<Element, Dim - 1, false>;
    friend class Tensor<Element, Dim + 1, true>;
    friend class Tensor<Element, Dim + 1, false>;

    template <typename... Sizes>
    auto set_sizes(dimension_index pos, size_type first, Sizes... rest) -> void
    {
        if (pos == 0) {
            _data_size = 1;
        }
        _dimensions[pos] = first;
        _data_size *= first;

        if constexpr (sizeof...(rest) > 0) {
            set_sizes(pos + 1, rest...);
        }
    }

    template <typename... Indices>
    auto get_index(dimension_index pos, size_type prev_index, size_type first, Indices... rest)
        -> size_type
    {
        size_type index = (prev_index * _dimensions[pos]) + first;
        if constexpr (sizeof...(rest) > 0) {
            return get_index(pos + 1, index, rest...);
        } else {
            return index;
        }
    }
};

}  // namespace ts
