#pragma once

namespace space {
template <typename DataType> class Tensor1D {
  private:
    DataType *_data;

  public:
    int length;

  public:
    explicit Tensor1D(int length) : _data(new DataType[length]), length(length) {}
    ~Tensor1D() { delete[] _data; }

    DataType & operator[](int index) {
        return _data[index];
    }
};
} // namespace space
