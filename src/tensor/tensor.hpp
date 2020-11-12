#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "dimensions.hpp"
#include "exceptions.hpp"
#include "iterator.hpp"
#include "ops.hpp"
#include "tensor_forward.hpp"


namespace ts {

/**
 *
 * @tparam Element is the type of array element
 * @tparam Dim is the number of dimensions
 */

template <typename Element, int Dim> class Tensor {

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
        _owner = true;
    }

    ~Tensor()
    {
        delete[] _dimensions;
        if (_owner) {
            delete _data;
        }
    }

    Tensor(std::initializer_list<Element> list)
    {
        _dimensions = new size_type[Dim];
        _owner = true;

        if (list.size() == 0) {
            _data_size = 0;
            _data = nullptr;
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
        _owner = true;

        if (list.size() == 0) {
            _data_size = 0;
            _data = nullptr;
            return;
        }

        size_type first_dim = list.size();
        size_type data_size = 0;
        for (Tensor<Element, Dim - 1> const &e : list) {
            if (data_size == 0) {
                _dimensions[0] = first_dim;
                std::copy(e.dimensions(), e.dimensions() + Dim - 1, _dimensions + 1);
            }
            data_size += e.data_size();
        }
        _data_size = data_size;
        _data = new Element[_data_size];

        Element *data_end = _data;
        for (Tensor<Element, Dim - 1> const &e : list) {
            data_end = std::copy(e.data(), e.data() + e.data_size(), data_end);
        }
    }

    Tensor(std::vector<Tensor<Element, Dim - 1>> list)
    {
        _dimensions = new size_type[Dim];
        _owner = true;
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
            data_end = std::copy(e.data(), e.data() + e.data_size(), data_end);
        }
    }

    explicit Tensor(std::vector<int> const & shape)
    {
        _owner = true;
        _dimensions = new size_type[Dim];
        std::copy(shape.begin(), shape.end(), _dimensions);
        _data_size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());
        _data = new Element[_data_size];
        for (size_type i = 0; i < _data_size; i++) {
            _data[i] = Element();
        }
    }

    template <typename... Sizes> Tensor(size_type first, Sizes... rest)
    {
        _owner = true;
        _dimensions = new size_type[Dim];
        set_sizes(0, first, rest...);
        _data = new Element[_data_size];
        for (size_type i = 0; i < _data_size; i++) {
            _data[i] = Element();
        }
    }

    Tensor(Dimensions const &sizes, Element *data)
    {
        _owner = false;
        _data = data;
        _dimensions = new size_type[Dim];
        std::copy(sizes.dimensions, sizes.dimensions + Dim, _dimensions);
        _data_size = sizes.data_size;
    }

    Tensor(Tensor const &tensor)
    {
        _owner = true;
        _data_size = tensor.data_size();
        _data = new Element[_data_size];
        _dimensions = new size_type[Dim];
        std::copy(tensor.data(), tensor.data() + tensor.data_size(), _data);
        std::copy(tensor.dimensions(), tensor.dimensions() + Dim, _dimensions);
    }

    // getting a reference to a subarray
    Tensor(Tensor<Element, Dim + 1> const & tensor, size_type index)
    {
        _dimensions = new size_type[Dim];
        std::copy(tensor.dimensions() + 1, tensor.dimensions() + Dim + 1, _dimensions);

        _data_size = tensor.data_size() / tensor.dimensions()[0];
        _data = tensor.data() + index * _data_size;
        _owner = false;
    }

    using iterator = IteratorClass<Element>;
    auto begin() -> iterator { return iterator::begin(data(), data_size()); }
    auto end() -> iterator { return iterator::end(data(), data_size()); }

    using const_iterator = IteratorClass<Element>;
    auto begin() const -> const_iterator { return iterator::begin(data(), data_size()); }
    auto end() const -> const_iterator { return iterator::end(data(), data_size()); }

    auto shape() const -> std::vector<int>
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
            return Tensor<Element, Dim - 1>(*this, first);
        } else if constexpr (Dim >= 2 && sizeof...(Indices) < Dim - 1) {
            return Tensor<Element, Dim - 1>(*this, first)(rest...);
        } else {
            throw TensorException("operator()");
        }
    }

    template <typename... Indices>
    auto operator()(size_type first, Indices... rest) const -> decltype(auto)
    {
        if constexpr (sizeof...(Indices) == Dim - 1) {
            return _data[get_index(0, 0, first, rest...)];
        } else if constexpr (Dim >= 2 && sizeof...(Indices) == 0) {
            return Tensor<Element, Dim - 1>(*this, first);
        } else if constexpr (Dim >= 2 && sizeof...(Indices) < Dim - 1) {
            return Tensor<Element, Dim - 1>(*this, first)(rest...);
        } else {
            throw TensorException("operator()");
        }
    }

    auto operator[](size_type i) -> decltype(auto)
    {
        if constexpr (Dim == 1) {
            return _data[i];
        } else {
            return Tensor<Element, Dim - 1>(*this, i);
        }
    }

    auto operator==(Tensor<Element, Dim> other) const -> bool
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

    auto operator!=(Tensor<Element, Dim> const &other) -> bool
    {
        return !(*this == other);
    }

    auto operator=(Tensor const &tensor) -> Tensor &
    {
        if(this == &tensor)
            return *this;

        if (_owner) {
            if (_data_size != tensor.data_size()) {
                std::ostringstream ss;
                ss << "operator= NON-OWNER CANNOT BE RESIZED. size1: " << _data_size
                   << " size2: " << tensor.data_size();
                throw TensorException(ss.str());
            }

            std::copy(tensor.data(), tensor.data() + _data_size, _data);
            std::copy(tensor.dimensions(), tensor.dimensions() + Dim, _dimensions);
        } else {
            _dimensions = tensor.dimensions();
            _data_size = tensor.data_size();
            _data = tensor.data();
        }
        return *this;
    }

    auto operator<(Element const &value) -> Tensor<bool, Dim>
    {
        return ts::mask<Element, Dim>(*this, [&](Element e) { return e < value; });
    }

    auto operator<=(Element const &value) -> Tensor<bool, Dim>
    {
        return ts::mask<Element, Dim>(*this, [&](Element e) { return e <= value; });
    }

    auto operator>(Element const &value) -> Tensor<bool, Dim>
    {
        return ts::mask<Element, Dim>(*this, [&](Element e) { return e > value; });
    }

    auto operator>=(Element const &value) -> Tensor<bool, Dim>
    {
        return ts::mask<Element, Dim>(*this, [&](Element e) { return e >= value; });
    }

    auto operator==(Element const &value) -> Tensor<bool, Dim>
    {
        return ts::mask<Element, Dim>(*this, [&](Element e) { return e == value; });
    }

    // TODO: is this code dead?
    template <typename... Sizes> auto resize(Sizes... sizes) -> void
    {
        if (_owner) {
            set_sizes(0, sizes...);
            delete[] _data;
            _data = new Element();
            for (int i = 0; i < _data_size; ++i) {
                _data[i] = Element();
            }
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
    auto get_index(dimension_index pos, size_type prev_index, size_type first, Indices... rest) const
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
