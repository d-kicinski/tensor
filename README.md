# tensor [![Documentation Status](https://readthedocs.org/projects/tensor-library/badge/?version=latest)](https://tensor-library.readthedocs.io/en/latest/?badge=latest)

| package | build/tests | coverage |
|---------|-------------|----------|
| C++    | [![Build C++](https://github.com/d-kicinski/tensor/actions/workflows/build.yml/badge.svg)](https://github.com/d-kicinski/tensor/actions/workflows/build.yml) | [![codecov](https://codecov.io/gh/d-kicinski/tensor/branch/master/graph/badge.svg?flag=cpp)](https://codecov.io/gh/d-kicinski/tensor) |
| Python | [![Build Python](https://github.com/d-kicinski/tensor/actions/workflows/python.yml/badge.svg)](https://github.com/d-kicinski/tensor/actions/workflows/python.yml) | [![codecov](https://codecov.io/gh/d-kicinski/tensor/branch/master/graph/badge.svg?flag=python)](https://codecov.io/gh/d-kicinski/tensor) |

--------------------------

This library provides a two main features:
- A class for interacting with multidimensional arrays (For backend library uses BLAS/LAPACK libraries with fallback to
own naive implementations).
- Deep neural networks.

The design goal is to create a numpy/pytorch alike interface for interacting
with multidimensional arrays packaged in a simple, relatively lightweight, library with limited external dependencies that
could be used on platforms like android phones and microcontrollers.



### How to use in your C++ project
If you're using cmake see [tensor-example](https://github.com/dawidkski/tensor-example) for example usage.

#### Usage of `tensor/nn` module
For example usage jump to [nn-planar-data example](https://github.com/d-kicinski/tensor/tree/master/examples/nn-planar-data)

#### Usage of `tensor` module in embedded application
See this [repository](https://github.com/d-kicinski/tensor-example-embedded)

#### Example usages of `tensor` module
For basic usages see this [doc]

### Usage of Python wrapper
Install the latest release `pip install https://github.com/d-kicinski/tensor/releases/download/v0.2.0/tensor-0.2.0-cp38-cp38-linux_x86_64.whl`.
Alternatively, clone this repo and build it by yourself.

The current state of autgrad capabilities can be seen in [here](https://github.com/d-kicinski/tensor/tree/master/python/examples/autograd)
