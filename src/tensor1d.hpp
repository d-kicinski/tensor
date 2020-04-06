#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>

namespace space {
template <typename DataType> class Tensor1D {

  private:
    DataType *_data{};

  public:
    size_t length{};

    explicit Tensor1D(size_t length) : _data(new DataType[length]), length(length) {}

    Tensor1D(std::initializer_list<DataType> list) : Tensor1D(list.size()) {
        int count = 0;
        for (auto element : list) {
            _data[count] = element;
            ++count;
        }
    }

    ~Tensor1D() { delete[] _data; }

    Tensor1D(Tensor1D const &) = delete; // to avoid shallow copies

    Tensor1D &operator=(Tensor1D const &) = delete; // to avoid shallow copies

    DataType &operator[](int index) { return _data[index]; }

    bool operator==(Tensor1D<DataType> const &tensor_r) {
        return std::equal(_data, std::end(_data), std::begin(tensor_r));
    }
};
} // namespace space
