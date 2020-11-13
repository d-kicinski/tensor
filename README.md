# tensor
[![Build Status](https://travis-ci.org/d-kicinski/tensor.svg?branch=master)](https://travis-ci.org/d-kicinski/tensor)
[![codecov](https://codecov.io/gh/d-kicinski/tensor/branch/master/graph/badge.svg)](https://codecov.io/gh/d-kicinski/tensor)


This library provides a class for interfacing with BLAS/LAPACK libraries with optional fallback to
own naive implementations. The design goal is to create a numpy alike interface for interacting
with multidimensional arrays packaged in a simple, relatively lightweight, library that
could be used on many different platforms like android phones and microcontrollers.

#### How to use
If you're using cmake see [tensor-example](https://github.com/dawidkski/tensor-example) for example usage.

#### Usage
Create multidimensional tensor:
```c++
// Create an array for storing bitmaps with variadic constructor
int constexpr width = 1024;
int constexpr height = 720;
int constexpr channels = 3;

ts::Tensor<int, 3> image(width, height, channels);
image.shape();      // std::vector{1024, 720, 3});
image.data_size();  // 1024 * 720 * 3 = 2211840
```

Multiply matrices:
```c++
// Construct 2D arrays via std::initializer_list
using Matrix = ts::Tensor<float, 2>;
Matrix A = {
    {3, 1, 3},
    {1, 5, 9},
};
Matrix B = {
    {3, 1},
    {1, 5},
    {2, 6}
};
// Multiply via free function
Matrix C = ts::dot(A, B);
// C = 
//    / 16 26 \
//    \ 26 80 /
```
