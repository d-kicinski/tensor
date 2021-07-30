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

Features:
* `tensor`:
    * classes and utilities for interacting with nd-arrays
    * sane interface to gemm routines
* `tensor/nn`: 
    * layers: FeedForward, Conv2D(naive/im2col), RNN, LSTM, Pooling
    * optimizers: SGD(with momentum), Adagrad, RMSProp, Adam
    * saving/restoring models using protobuf
* Python
    * wrapper for major of the `tensor` and `tensor/nn` functionalities
    * experimental autograd module (PyTorch alike)
    
Coming soon:
* `tensor`:
    * improved naive GEMM implementation using AVX2 intrinsics
* `tensor/nn`:
    * layers: SelfAttention, BatchNorm, LayerNorm


### Examples
* C++:
    * [nn-planar-data example](https://github.com/d-kicinski/tensor/tree/master/examples/nn-planar-data)
    * [char-rnn example](https://github.com/d-kicinski/tensor/tree/master/examples/char-rnn)
    * [embedded application](https://github.com/d-kicinski/tensor-example-embedded)
    
* Python:
    * [MNIST](https://github.com/d-kicinski/tensor/tree/master/python/examples/mnist)
    * [autograd capabilities](https://github.com/d-kicinski/tensor/tree/master/python/examples/autograd)
  
### How to use in your C++ project
If you're using cmake see [tensor-example](https://github.com/dawidkski/tensor-example) for example usage.

### How to use Python wrapper
Install the latest release `pip install https://github.com/d-kicinski/tensor/releases/download/v0.2.0/tensor-0.2.0-cp38-cp38-linux_x86_64.whl`.
Alternatively, clone this repo and build it by yourself.


### Documentation
For basic usages see this [doc](https://tensor-library.readthedocs.io/en/latest/usage.html)

