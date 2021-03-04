Basic usage
============

**Create multidimensional tensor:**

.. code-block:: c++

    // Create an array for storing bitmaps with variadic constructor
    int constexpr width = 1024;
    int constexpr height = 720;
    int constexpr channels = 3;

    ts::Tensor<int, 3> image(width, height, channels);
    image.shape();      // std::array{1024, 720, 3});
    image.data_size();  // 1024 * 720 * 3 = 2211840

**Multiply matrices:**

.. code-block:: c++

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